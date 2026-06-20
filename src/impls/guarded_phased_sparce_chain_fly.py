import math
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
)
from .phased_sparce_chain_fly import (
    PREFIX,
    create_protocol as create_base_protocol,
)

# Same sender as phased_sparce_chain_fly.
# Receiver policy:
# - recover phase from an alternating-run model;
# - treat same observed phase as a soft boundary, because it is often a missed
#   deletion or delimiter under skip_observation < 1;
# - only feed locally clean windows into peeling. This keeps asymptotic decode
#   cost in the same class as the base implementation.

COMMIT_LAG = 1
MIN_PHASE_MARGIN = 2


def create_protocol(config: Config) -> Protocol:
    base_protocol = create_base_protocol(config)

    packet_bitsize = int(config.packet_bitsize)
    message_bitsize = int(config.message_bitsize)

    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2 to reserve a phase bit")
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    z = packet_bitsize - 1
    symbol_q = 1 << z
    symbol_mask = symbol_q - 1

    input_bits = PREFIX * z
    if input_bits > MAX_PREFIX_INPUT_BITS:
        raise ValueError(
            "guarded_phased_sparce_chain_fly is tailored for small packet sizes: "
            f"PREFIX*(packet_bitsize-1) must be <= {MAX_PREFIX_INPUT_BITS}, got {input_bits}"
        )

    max_bits = max_message_bitsize(packet_bitsize)
    if message_bitsize > max_bits:
        raise ValueError(
            "message_bitsize too large for guarded_phased_sparce_chain_fly: "
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
        seen_a: Dict[int, int] = {}
        seen_b: Dict[int, int] = {}

        run_symbols: List[int] = []
        run_observed_phases: List[bool] = []
        next_edge_index = PREFIX
        total_k = k1 + k2

        def progress() -> float:
            if total_k == 0:
                return 1.0
            known = sum(1 for v in symbols_a if v is not None) + sum(
                1 for v in symbols_b if v is not None
            )
            return known / total_k

        def maybe_done() -> np.ndarray | None:
            if any(v is None for v in symbols_a) or any(v is None for v in symbols_b):
                return None

            parts = []
            if k1 > 0:
                vec_a = np.array(symbols_a, dtype=np.uint16)
                parts.append(message_from_message_vector(vec_a, half_bits_a, symbol_q))
            else:
                parts.append(np.zeros(0, dtype=np.bool_))

            if k2 > 0:
                vec_b = np.array(symbols_b, dtype=np.uint16)
                parts.append(message_from_message_vector(vec_b, half_bits_b, symbol_q))
            else:
                parts.append(np.zeros(0, dtype=np.bool_))

            return np.concatenate(parts)

        def phase_model() -> tuple[bool, int]:
            true_votes = sum(
                int(bool(phase) ^ bool(i & 1))
                for i, phase in enumerate(run_observed_phases)
            )
            false_votes = len(run_observed_phases) - true_votes
            if true_votes > false_votes:
                return True, true_votes - false_votes
            return False, false_votes - true_votes

        def model_phase(start_phase: bool, index: int) -> bool:
            return bool(start_phase ^ bool(index & 1))

        def reset_run(symbol: int | None = None, phase: bool | None = None) -> None:
            nonlocal next_edge_index
            run_symbols.clear()
            run_observed_phases.clear()
            next_edge_index = PREFIX
            if symbol is not None and phase is not None:
                run_symbols.append(symbol)
                run_observed_phases.append(phase)

        def add_equation(x: int, y: int, src_phase: bool, dst_phase: bool) -> None:
            if (not src_phase) and dst_phase:
                if k1 > 0 and x not in seen_a:
                    seen_a[x] = y
                    peel_add_equation(x, y, k1, cdf_a, salt_a, symbols_a, pending_a)
                    peel_propagate(symbols_a, pending_a)
            elif src_phase and (not dst_phase):
                if k2 > 0 and x not in seen_b:
                    seen_b[x] = y
                    peel_add_equation(x, y, k2, cdf_b, salt_b, symbols_b, pending_b)
                    peel_propagate(symbols_b, pending_b)

        def commit_ready_edges() -> None:
            nonlocal next_edge_index
            if len(run_symbols) < PREFIX + 1:
                return

            start_phase, margin = phase_model()
            if margin < MIN_PHASE_MARGIN:
                return

            max_ready = len(run_symbols) - 1 - COMMIT_LAG
            while next_edge_index <= max_ready:
                window_start = next_edge_index - PREFIX
                window_end = next_edge_index + 1
                locally_clean = all(
                    run_observed_phases[idx] == model_phase(start_phase, idx)
                    for idx in range(window_start, window_end)
                )
                if locally_clean:
                    x = pack_prefix_symbols(
                        run_symbols[next_edge_index - PREFIX : next_edge_index], z
                    )
                    y = run_symbols[next_edge_index]
                    add_equation(
                        x,
                        y,
                        model_phase(start_phase, next_edge_index - 1),
                        model_phase(start_phase, next_edge_index),
                    )
                next_edge_index += 1

        while True:
            done = maybe_done()
            if done is not None:
                return done

            packet = yield progress()

            if packet is None:
                reset_run()
                continue

            curr_phase = bool(packet[0])
            curr_symbol = int(bool_array_to_uint16(packet) & symbol_mask)

            if run_observed_phases and curr_phase == run_observed_phases[-1]:
                reset_run(curr_symbol, curr_phase)
                continue

            run_symbols.append(curr_symbol)
            run_observed_phases.append(curr_phase)
            commit_ready_edges()

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
