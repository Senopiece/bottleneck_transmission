import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol
from ._utils.conversions import bool_array_to_uint16, message_from_message_vector
from ._utils.sparce import (
    MAX_PREFIX_INPUT_BITS,
    pack_prefix_symbols,
    peel_add_equation,
    peel_propagate,
    robust_soliton_cdf,
    subset_from_x,
)
from .phased_sparce_chain_fly import (
    PREFIX,
    create_protocol as create_base_protocol,
)

# Same sender as phased_sparce_chain_fly, but a more conservative receiver:
# - infer phase from alternating-run majority instead of trusting one phase bit;
# - keep low-confidence observations out of peeling;
# - validate/repair the final solution against all collected equations.

COMMIT_LAG = 1
MIN_PEEL_WEIGHT = 1.0
REPAIR_PASSES = 3
VALIDATION_EVENTS = 4


def create_protocol(config: Config) -> Protocol:
    base_protocol = create_base_protocol(config)

    packet_bitsize = int(config.packet_bitsize)
    message_bitsize = int(config.message_bitsize)

    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2 to reserve a phase bit")
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    N = packet_bitsize
    z = N - 1
    symbol_q = 1 << z
    symbol_mask = symbol_q - 1

    input_bits = PREFIX * z
    if input_bits > MAX_PREFIX_INPUT_BITS:
        raise ValueError(
            "reliable_phased_sparce_chain_fly is tailored for small packet sizes: "
            f"PREFIX*(packet_bitsize-1) must be <= {MAX_PREFIX_INPUT_BITS}, got {input_bits}"
        )

    max_bits = max_message_bitsize(packet_bitsize)
    if message_bitsize > max_bits:
        raise ValueError(
            "message_bitsize too large for reliable_phased_sparce_chain_fly: "
            f"max {max_bits} for packet_bitsize={packet_bitsize}"
        )

    half_bits_a = (message_bitsize + 1) // 2
    half_bits_b = message_bitsize - half_bits_a

    k1 = math.ceil(half_bits_a / z) if half_bits_a > 0 else 0
    k2 = math.ceil(half_bits_b / z) if half_bits_b > 0 else 0

    cdf_a = robust_soliton_cdf(k1) if k1 > 0 else [1.0]
    cdf_b = robust_soliton_cdf(k2) if k2 > 0 else [1.0]

    salt_a = 0x9E3779B97F4A7C15
    salt_b = 0xD1B54A32D192ED03

    def make_estimator() -> Estimator:
        symbols_a: List[int | None] = [None for _ in range(k1)]
        symbols_b: List[int | None] = [None for _ in range(k2)]
        pending_a: List[Tuple[Set[int], int]] = []
        pending_b: List[Tuple[Set[int], int]] = []

        peeled_x_a: Set[int] = set()
        peeled_x_b: Set[int] = set()

        # Equations are (subset, rhs, confidence_weight).
        equations_a: List[Tuple[List[int], int, float]] = []
        equations_b: List[Tuple[List[int], int, float]] = []
        observed_a: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        observed_b: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        run_symbols: List[int] = []
        run_observed_phases: List[bool] = []
        next_edge_index = PREFIX
        validation_events_remaining: int | None = None

        total_k = k1 + k2

        def progress() -> float:
            if total_k == 0:
                return 1.0
            known = sum(1 for v in symbols_a if v is not None) + sum(
                1 for v in symbols_b if v is not None
            )
            return known / total_k

        def phase_model(phases: List[bool]) -> tuple[bool, int, int]:
            if not phases:
                return False, 0, 0

            true_votes = sum(int(bool(phase) ^ bool(i & 1)) for i, phase in enumerate(phases))
            false_votes = len(phases) - true_votes
            if true_votes > false_votes:
                return True, true_votes, true_votes - false_votes
            return False, false_votes, false_votes - true_votes

        def model_phase(start_phase: bool, index: int) -> bool:
            return bool(start_phase ^ bool(index & 1))

        def local_phase_weight(start_phase: bool, edge_index: int, margin: int) -> float:
            start = edge_index - PREFIX
            end = edge_index + 1
            errors = 0
            for idx in range(start, end):
                if run_observed_phases[idx] != model_phase(start_phase, idx):
                    errors += 1

            if errors == 0 and margin >= 2:
                return 1.0
            if errors == 0:
                return 0.8
            if errors == 1:
                return 0.1
            return 0.0

        def add_equation(
            x: int,
            y: int,
            src_phase: bool,
            dst_phase: bool,
            weight: float,
        ) -> None:
            if weight <= 0.0:
                return

            if (not src_phase) and dst_phase:
                if k1 <= 0:
                    return
                subset = subset_from_x(x, k1, cdf_a, salt_a, singleton_limit=k1)
                equations_a.append((subset, y, weight))
                observed_a[x][y] += weight
                if weight >= MIN_PEEL_WEIGHT and x not in peeled_x_a:
                    y_majority = max(observed_a[x].items(), key=lambda item: item[1])[0]
                    peeled_x_a.add(x)
                    peel_add_equation(
                        x, y_majority, k1, cdf_a, salt_a, symbols_a, pending_a
                    )
                    peel_propagate(symbols_a, pending_a)
            elif src_phase and (not dst_phase):
                if k2 <= 0:
                    return
                subset = subset_from_x(x, k2, cdf_b, salt_b, singleton_limit=k2)
                equations_b.append((subset, y, weight))
                observed_b[x][y] += weight
                if weight >= MIN_PEEL_WEIGHT and x not in peeled_x_b:
                    y_majority = max(observed_b[x].items(), key=lambda item: item[1])[0]
                    peeled_x_b.add(x)
                    peel_add_equation(
                        x, y_majority, k2, cdf_b, salt_b, symbols_b, pending_b
                    )
                    peel_propagate(symbols_b, pending_b)

        def commit_ready_edges() -> None:
            nonlocal next_edge_index
            if len(run_symbols) < PREFIX + 1:
                return

            start_phase, _score, margin = phase_model(run_observed_phases)
            max_ready = len(run_symbols) - 1 - COMMIT_LAG

            while next_edge_index <= max_ready:
                x = pack_prefix_symbols(
                    run_symbols[next_edge_index - PREFIX : next_edge_index], z
                )
                y = run_symbols[next_edge_index]
                src_phase = model_phase(start_phase, next_edge_index - 1)
                dst_phase = model_phase(start_phase, next_edge_index)
                weight = local_phase_weight(start_phase, next_edge_index, margin)
                add_equation(x, y, src_phase, dst_phase, weight)
                next_edge_index += 1

        def reset_run(symbol: int | None = None, phase: bool | None = None) -> None:
            nonlocal next_edge_index
            run_symbols.clear()
            run_observed_phases.clear()
            next_edge_index = PREFIX
            if symbol is not None and phase is not None:
                run_symbols.append(symbol)
                run_observed_phases.append(phase)

        def domain_penalty(
            values: List[int], equations: List[Tuple[List[int], int, float]]
        ) -> float:
            penalty = 0.0
            for subset, rhs, weight in equations:
                lhs = 0
                for idx in subset:
                    lhs ^= values[idx]
                if lhs != rhs:
                    penalty += weight
            return penalty

        def repair_domain(
            symbols: List[int | None],
            equations: List[Tuple[List[int], int, float]],
        ) -> tuple[List[int], float]:
            values = [int(v) for v in symbols if v is not None]
            if len(values) != len(symbols) or not equations:
                return values, 0.0

            by_symbol: List[List[Tuple[List[int], int, float]]] = [
                [] for _ in range(len(values))
            ]
            for equation in equations:
                subset, _rhs, _weight = equation
                for idx in subset:
                    by_symbol[idx].append(equation)

            def local_penalty(idx: int, candidate: int) -> float:
                old = values[idx]
                values[idx] = candidate
                penalty = 0.0
                for subset, rhs, weight in by_symbol[idx]:
                    lhs = 0
                    for symbol_idx in subset:
                        lhs ^= values[symbol_idx]
                    if lhs != rhs:
                        penalty += weight
                values[idx] = old
                return penalty

            for _ in range(REPAIR_PASSES):
                changed = False
                for idx in range(len(values)):
                    if not by_symbol[idx]:
                        continue
                    current = values[idx]
                    best_value = current
                    best_penalty = local_penalty(idx, current)
                    for candidate in range(symbol_q):
                        if candidate == current:
                            continue
                        candidate_penalty = local_penalty(idx, candidate)
                        if candidate_penalty + 1e-12 < best_penalty:
                            best_penalty = candidate_penalty
                            best_value = candidate
                    if best_value != current:
                        values[idx] = best_value
                        changed = True
                if not changed:
                    break

            return values, domain_penalty(values, equations)

        def maybe_done() -> np.ndarray | None:
            nonlocal validation_events_remaining
            if any(v is None for v in symbols_a) or any(v is None for v in symbols_b):
                return None

            repaired_a, _penalty_a = repair_domain(symbols_a, equations_a)
            repaired_b, _penalty_b = repair_domain(symbols_b, equations_b)
            total_penalty = _penalty_a + _penalty_b

            if total_penalty > 0.0:
                if validation_events_remaining is None:
                    validation_events_remaining = VALIDATION_EVENTS
                    return None
                if validation_events_remaining > 0:
                    return None

            parts = []
            if k1 > 0:
                vec_a = np.array(repaired_a, dtype=np.uint16)
                parts.append(message_from_message_vector(vec_a, half_bits_a, symbol_q))
            else:
                parts.append(np.zeros(0, dtype=np.bool_))

            if k2 > 0:
                vec_b = np.array(repaired_b, dtype=np.uint16)
                parts.append(message_from_message_vector(vec_b, half_bits_b, symbol_q))
            else:
                parts.append(np.zeros(0, dtype=np.bool_))

            return np.concatenate(parts)

        while True:
            done = maybe_done()
            if done is not None:
                return done

            packet = yield progress()

            if packet is None:
                reset_run()
                if validation_events_remaining is not None and validation_events_remaining > 0:
                    validation_events_remaining -= 1
                continue

            curr_phase = bool(packet[0])
            curr_symbol = int(bool_array_to_uint16(packet) & symbol_mask)

            if (
                run_symbols
                and curr_phase == run_observed_phases[-1]
                and curr_symbol == run_symbols[-1]
            ):
                reset_run(curr_symbol, curr_phase)
                continue

            run_symbols.append(curr_symbol)
            run_observed_phases.append(curr_phase)
            commit_ready_edges()
            if validation_events_remaining is not None and validation_events_remaining > 0:
                validation_events_remaining -= 1

    return Protocol(
        make_sampler=base_protocol.make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if packet_bitsize <= 1:
        return 0
    z = packet_bitsize - 1
    input_bits = PREFIX * z
    if input_bits > MAX_PREFIX_INPUT_BITS:
        return 0
    return 2 * z * (1 << input_bits)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
