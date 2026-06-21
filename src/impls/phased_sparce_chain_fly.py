import math
from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.sparce import (
    MAX_PREFIX_INPUT_BITS,
    pack_prefix_symbols,
    peel_add_equation,
    peel_propagate,
    robust_soliton_cdf,
    subset_from_x,
)

# Domain:
# skip_probability: [0, 1)
# corruption_probability: 0
# skip_observation: 1.0

SEGMENT_PROBES = 4
RECENT_WINDOW_MARGIN = 6
SEGMENT_LENGTH_BONUS = 0.0
PHASE_MIXING = False
PHASE_MIX_MODE = "parity"
PHASE_MIX_MASK = 0xFFFF
PHASE_SYMBOL_XOR_MASK = 0
PHASE_MATRIX_INV_COLUMNS = (0b10011, 0b11100, 0b00101, 0b01010, 0b00111)
PHASE_REPAIR = False
PHASE_CANDIDATE_DECODING = False

# Number of previous (n-1)-bit symbols packed into x for f(x).
PREFIX = 2


def sampler_seed(
    symbols_a: np.ndarray,
    symbols_b: np.ndarray,
    packet_bitsize: int,
    message_bitsize: int,
) -> int:
    seed = 1469598103934665603
    fnv_prime = 1099511628211

    seed ^= packet_bitsize & 0xFFFF
    seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    seed ^= message_bitsize & 0xFFFFFFFF
    seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF

    for coeff in symbols_a:
        seed ^= int(coeff) + 1
        seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    seed ^= 0x9E3779B9
    seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    for coeff in symbols_b:
        seed ^= int(coeff) + 1
        seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF

    return seed


def create_protocol(config: Config) -> Protocol:
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
            "phased_sparce_chain_fly is tailored for small packet sizes: "
            f"PREFIX*(packet_bitsize-1) must be <= {MAX_PREFIX_INPUT_BITS}, got {input_bits}"
        )
    state_q = 1 << input_bits
    state_mask = state_q - 1

    max_bits = max_message_bitsize(packet_bitsize)
    if message_bitsize > max_bits:
        raise ValueError(
            f"message_bitsize too large for phased_sparce_chain_fly: max {max_bits} for packet_bitsize={packet_bitsize}"
        )

    half_bits_a = (message_bitsize + 1) // 2
    half_bits_b = message_bitsize - half_bits_a

    k1 = math.ceil(half_bits_a / z) if half_bits_a > 0 else 0
    k2 = math.ceil(half_bits_b / z) if half_bits_b > 0 else 0

    cdf_a = robust_soliton_cdf(k1) if k1 > 0 else [1.0]
    cdf_b = robust_soliton_cdf(k2) if k2 > 0 else [1.0]

    salt_a = 0x9E3779B97F4A7C15
    salt_b = 0xD1B54A32D192ED03

    def make_sampler(message: Message) -> Sampler:
        msg_a = message[:half_bits_a]
        msg_b = message[half_bits_a:]

        symbols_a = (
            make_message_vector(msg_a, k1, symbol_q)
            if k1 > 0
            else np.empty(0, dtype=np.uint16)
        )
        symbols_b = (
            make_message_vector(msg_b, k2, symbol_q)
            if k2 > 0
            else np.empty(0, dtype=np.uint16)
        )

        def f1(x: int) -> int:
            if k1 == 0:
                return 0
            subset = subset_from_x(x, k1, cdf_a, salt_a, singleton_limit=k1)
            y = 0
            for idx in subset:
                y ^= int(symbols_a[idx])
            return y & symbol_mask

        def f2(x: int) -> int:
            if k2 == 0:
                return 0
            subset = subset_from_x(x, k2, cdf_b, salt_b, singleton_limit=k2)
            y = 0
            for idx in subset:
                y ^= int(symbols_b[idx])
            return y & symbol_mask

        def phase_mix_mode() -> str:
            return PHASE_MIX_MODE if PHASE_MIXING else "none"

        def phase_mix_bit(symbol: int) -> bool:
            if phase_mix_mode() == "none":
                return False
            value = int(symbol) & PHASE_MIX_MASK
            value ^= value >> 2
            value ^= value >> 1
            return bool(value & 1)

        def gray_encode(value: int) -> int:
            return int(value) ^ (int(value) >> 1)

        def gray_decode(value: int) -> int:
            result = int(value)
            shift = 1
            while shift < N:
                result ^= result >> shift
                shift <<= 1
            return result

        def matrix_decode(value: int) -> int:
            result = 0
            columns = PHASE_MATRIX_INV_COLUMNS
            if len(columns) != N:
                raise ValueError("PHASE_MATRIX_INV_COLUMNS length must equal packet_bitsize")
            for bit in range(N):
                if int(value) & (1 << bit):
                    result ^= int(columns[bit])
            return result & ((1 << N) - 1)

        matrix_decode_table: list[int] | None = None
        matrix_encode_table: list[int] | None = None
        if phase_mix_mode() == "matrix":
            matrix_decode_table = [matrix_decode(value) for value in range(1 << N)]
            matrix_encode_table = [0 for _ in range(1 << N)]
            for physical_value, logical_value in enumerate(matrix_decode_table):
                matrix_encode_table[logical_value] = physical_value
            if len(set(matrix_decode_table)) != (1 << N):
                raise ValueError("PHASE_MATRIX_INV_COLUMNS must define an invertible matrix")

        def packet_id(symbol: int, phase_b: bool) -> int:
            mode = phase_mix_mode()
            symbol = int(symbol) & symbol_mask
            phase_b = bool(phase_b)
            if mode == "matrix":
                assert matrix_encode_table is not None
                return matrix_encode_table[symbol + (symbol_q if phase_b else 0)]
            if mode == "gray":
                return gray_encode(symbol + (symbol_q if phase_b else 0))

            physical_symbol = symbol
            if mode in ("symbol_xor", "affine") and phase_b:
                physical_symbol ^= PHASE_SYMBOL_XOR_MASK & symbol_mask

            physical_phase = phase_b
            if mode in ("parity", "affine"):
                physical_phase ^= phase_mix_bit(physical_symbol)

            return int(physical_symbol) + (symbol_q if physical_phase else 0)

        def packet_from_id(value: int) -> tuple[int, bool]:
            mode = phase_mix_mode()
            value = int(value)
            if mode == "matrix":
                assert matrix_decode_table is not None
                decoded = matrix_decode_table[value]
                return decoded & symbol_mask, bool(decoded & symbol_q)
            if mode == "gray":
                decoded = gray_decode(value)
                return decoded & symbol_mask, bool(decoded & symbol_q)

            physical_symbol = value & symbol_mask
            physical_phase = bool(value & symbol_q)

            phase_b = physical_phase
            if mode in ("parity", "affine"):
                phase_b ^= phase_mix_bit(physical_symbol)

            symbol = physical_symbol
            if mode in ("symbol_xor", "affine") and phase_b:
                symbol ^= PHASE_SYMBOL_XOR_MASK & symbol_mask

            return symbol, phase_b

        def phased_output(symbol: int, is_phase_b: bool):
            return uint16_to_bool_array(np.uint16(packet_id(symbol, is_phase_b)), N)

        def state_id(x: int, src_phase_b: bool) -> int:
            return int(x) + (state_q if src_phase_b else 0)

        def state_from_id(value: int) -> tuple[int, bool]:
            return value & state_mask, bool(value & state_q)

        def x_to_symbols(x: int) -> list[int]:
            symbols = [0 for _ in range(PREFIX)]
            curr = int(x)
            for i in range(PREFIX - 1, -1, -1):
                symbols[i] = curr & symbol_mask
                curr >>= z
            return symbols

        prefix_tail_mask = (1 << ((PREFIX - 1) * z)) - 1 if PREFIX > 1 else 0

        def shift_x(x: int, y: int) -> int:
            return (((int(x) & prefix_tail_mask) << z) | int(y)) & state_mask

        def prefix_packets_for_state(sid: int) -> list[int]:
            x, src_phase_b = state_from_id(sid)
            symbols = x_to_symbols(x)
            return [
                packet_id(symbol, bool(src_phase_b ^ ((PREFIX - 1 - i) & 1)))
                for i, symbol in enumerate(symbols)
            ]

        state_count = 2 * state_q
        nxt_state = np.empty(state_count, dtype=np.int32)
        nxt_packet = np.empty(state_count, dtype=np.int32)
        for sid in range(state_count):
            x, src_phase_b = state_from_id(sid)
            next_symbol = f2(x) if src_phase_b else f1(x)
            next_phase_b = not src_phase_b
            nxt_packet[sid] = packet_id(next_symbol, next_phase_b)
            nxt_state[sid] = state_id(shift_x(x, next_symbol), next_phase_b)

        max_window = max(0, symbol_q - RECENT_WINDOW_MARGIN)
        window_a = min(max_window, k1)
        window_b = min(max_window, k2)
        recent_counts_a = np.zeros(state_count, dtype=np.int32)
        recent_counts_b = np.zeros(state_count, dtype=np.int32)
        recent_a: deque[int] = deque()
        recent_b: deque[int] = deque()

        probes = min(SEGMENT_PROBES, state_count)
        rng = np.random.default_rng(
            sampler_seed(symbols_a, symbols_b, packet_bitsize, message_bitsize)
        )
        singleton_emit_counts = np.zeros(state_count, dtype=np.int32)

        singleton_ids_a = np.arange(0, min(k1, state_q), dtype=np.int32)
        singleton_ids_b = np.arange(
            state_q, state_q + min(k2, state_q), dtype=np.int32
        )
        all_singleton_ids = np.concatenate((singleton_ids_a, singleton_ids_b))

        def counts_for_phase(phase_b: bool) -> np.ndarray:
            return recent_counts_b if phase_b else recent_counts_a

        def recent_for_phase(phase_b: bool) -> deque[int]:
            return recent_b if phase_b else recent_a

        def window_for_phase(phase_b: bool) -> int:
            return window_b if phase_b else window_a

        def singleton_ids_for_phase(phase_b: bool) -> np.ndarray:
            return singleton_ids_b if phase_b else singleton_ids_a

        def is_singleton_input(sid: int) -> bool:
            x, src_phase_b = state_from_id(sid)
            return x < (k2 if src_phase_b else k1)

        def remember(sid: int) -> None:
            _, src_phase_b = state_from_id(sid)
            window = window_for_phase(src_phase_b)
            if window <= 0:
                return
            recent_counts = counts_for_phase(src_phase_b)
            recent = recent_for_phase(src_phase_b)
            recent.append(sid)
            recent_counts[sid] += 1
            if len(recent) > window:
                dropped = recent.popleft()
                recent_counts[dropped] -= 1

        def build_virtual_segment(start: int) -> list[int]:
            visited = {start}
            segment = [start]
            cur = start

            while True:
                next_id = int(nxt_state[cur])
                segment.append(next_id)
                _, next_phase_b = state_from_id(next_id)
                if counts_for_phase(next_phase_b)[next_id] > 0 or next_id in visited:
                    return segment
                visited.add(next_id)
                cur = next_id

        def choose_probe_starts(start_phase_b: bool) -> np.ndarray:
            phase_offset = state_q if start_phase_b else 0
            singleton_ids = singleton_ids_for_phase(start_phase_b)
            if singleton_ids.size > 0:
                min_singleton_count = int(np.min(singleton_emit_counts[singleton_ids]))
                singleton_starts = singleton_ids[
                    singleton_emit_counts[singleton_ids] == min_singleton_count
                ]
                eligible_singletons = singleton_starts[
                    counts_for_phase(start_phase_b)[singleton_starts] == 0
                ]
                if eligible_singletons.size == 0:
                    eligible_singletons = singleton_starts

                if eligible_singletons.size > 0:
                    probe_count = min(probes, int(eligible_singletons.size))
                    return rng.choice(
                        eligible_singletons.astype(np.int32),
                        size=probe_count,
                        replace=False,
                    )

            phase_ids = np.arange(phase_offset, phase_offset + state_q, dtype=np.int32)
            recent_counts = counts_for_phase(start_phase_b)
            eligible = phase_ids[recent_counts[phase_ids] == 0]
            if eligible.size == 0:
                eligible = phase_ids

            probe_count = min(probes, int(eligible.size))
            if probe_count <= 0:
                probe_count = 1

            return rng.choice(eligible, size=probe_count, replace=False)

        def has_pending_singletons(phase_b: bool) -> bool:
            singleton_ids = singleton_ids_for_phase(phase_b)
            if singleton_ids.size == 0:
                return False
            return bool(np.any(singleton_emit_counts[singleton_ids] == 0))

        def segment_end_phase(segment: list[int]) -> bool:
            _, phase_b = state_from_id(segment[-1])
            return phase_b

        def segment_score(segment: list[int]) -> tuple[int, float, int]:
            if all_singleton_ids.size == 0:
                return (0, 0.0, len(segment))

            min_count = int(np.min(singleton_emit_counts[all_singleton_ids]))
            rare_edges = 0
            weighted_freshness = 0.0

            for curr_state, _next_state in zip(segment, segment[1:]):
                if not is_singleton_input(curr_state):
                    continue
                count = int(singleton_emit_counts[curr_state])
                if count == min_count:
                    rare_edges += 1
                weighted_freshness += 1.0 / float(count + 1)

            if SEGMENT_LENGTH_BONUS:
                weighted_freshness += SEGMENT_LENGTH_BONUS * len(segment)

            return (rare_edges, weighted_freshness, len(segment))

        next_start_phase_b = False
        while True:
            starts = choose_probe_starts(next_start_phase_b)

            best_segment: list[int] = []
            best_score: tuple[int, float, int] | None = None
            for start in starts:
                candidate = build_virtual_segment(int(start))
                candidate_score = segment_score(candidate)
                if best_score is None or candidate_score > best_score:
                    best_segment = candidate
                    best_score = candidate_score

            if not best_segment:
                start_id = state_id(0, next_start_phase_b)
                best_segment = [start_id, int(nxt_state[start_id])]

            if has_pending_singletons(next_start_phase_b):
                desired_next_phase_b = next_start_phase_b
            elif has_pending_singletons(not next_start_phase_b):
                desired_next_phase_b = not next_start_phase_b
            else:
                desired_next_phase_b = segment_end_phase(best_segment)

            if (
                len(best_segment) > 2
                and segment_end_phase(best_segment) != desired_next_phase_b
            ):
                best_segment.pop()

            prefix_packets = prefix_packets_for_state(best_segment[0])
            first_symbol, first_phase_b = packet_from_id(prefix_packets[0])
            yield phased_output(first_symbol, first_phase_b)
            for pid in prefix_packets:
                symbol, phase_b = packet_from_id(pid)
                yield phased_output(symbol, phase_b)

            for curr_state, _next_state in zip(best_segment, best_segment[1:]):
                if is_singleton_input(curr_state):
                    singleton_emit_counts[curr_state] += 1

            for sid in best_segment[:-1]:
                remember(sid)
                pid = int(nxt_packet[sid])
                symbol, phase_b = packet_from_id(pid)
                yield phased_output(symbol, phase_b)

            _, next_start_phase_b = state_from_id(best_segment[-1])

    def make_estimator() -> Estimator:
        symbols_a: List[int | None] = [None for _ in range(k1)]
        symbols_b: List[int | None] = [None for _ in range(k2)]
        pending_a: List[Tuple[Set[int], int]] = []
        pending_b: List[Tuple[Set[int], int]] = []

        seen_a: Dict[int, int] = {}
        seen_b: Dict[int, int] = {}

        run_symbols: List[int] = []
        run_phases: List[bool] = []
        max_run_keep = PREFIX + 1
        pending_edge: Tuple[int, int, bool, bool] | None = None

        total_k = k1 + k2

        def observed_phase_mix_mode() -> str:
            return PHASE_MIX_MODE if PHASE_MIXING else "none"

        def observed_phase_mix_bit(symbol: int) -> bool:
            if observed_phase_mix_mode() == "none":
                return False
            value = int(symbol) & PHASE_MIX_MASK
            value ^= value >> 2
            value ^= value >> 1
            return bool(value & 1)

        def observed_gray_decode(value: int) -> int:
            result = int(value)
            shift = 1
            while shift < N:
                result ^= result >> shift
                shift <<= 1
            return result

        def observed_matrix_decode(value: int) -> int:
            result = 0
            columns = PHASE_MATRIX_INV_COLUMNS
            if len(columns) != N:
                raise ValueError("PHASE_MATRIX_INV_COLUMNS length must equal packet_bitsize")
            for bit in range(N):
                if int(value) & (1 << bit):
                    result ^= int(columns[bit])
            return result & ((1 << N) - 1)

        observed_matrix_decode_table: list[int] | None = None
        if observed_phase_mix_mode() == "matrix":
            observed_matrix_decode_table = [
                observed_matrix_decode(value) for value in range(1 << N)
            ]
            if len(set(observed_matrix_decode_table)) != (1 << N):
                raise ValueError("PHASE_MATRIX_INV_COLUMNS must define an invertible matrix")

        def observed_packet_from_id(value: int) -> tuple[int, bool]:
            mode = observed_phase_mix_mode()
            value = int(value)
            if mode == "matrix":
                assert observed_matrix_decode_table is not None
                decoded = observed_matrix_decode_table[value]
                return decoded & symbol_mask, bool(decoded & symbol_q)
            if mode == "gray":
                decoded = observed_gray_decode(value)
                return decoded & symbol_mask, bool(decoded & symbol_q)

            physical_symbol = value & symbol_mask
            physical_phase = bool(value & symbol_q)

            phase_b = physical_phase
            if mode in ("parity", "affine"):
                phase_b ^= observed_phase_mix_bit(physical_symbol)

            symbol = physical_symbol
            if mode in ("symbol_xor", "affine") and phase_b:
                symbol ^= PHASE_SYMBOL_XOR_MASK & symbol_mask

            return symbol, phase_b

        def observed_packet_candidates(value: int) -> list[tuple[int, bool, int]]:
            symbol, phase_b = observed_packet_from_id(value)
            if not PHASE_CANDIDATE_DECODING:
                return [(symbol, phase_b, 0)]

            candidates: dict[tuple[int, bool], int] = {(symbol, phase_b): 0}
            for bit in range(N):
                candidate_value = int(value) ^ (1 << bit)
                candidate = observed_packet_from_id(candidate_value)
                if candidate not in candidates:
                    candidates[candidate] = 1

            return [
                (candidate_symbol, candidate_phase, penalty)
                for (candidate_symbol, candidate_phase), penalty in candidates.items()
            ]

        def choose_observed_packet(value: int) -> tuple[int, bool]:
            candidates = observed_packet_candidates(value)
            raw_symbol, raw_phase, _raw_penalty = candidates[0]
            if not PHASE_CANDIDATE_DECODING or not run_phases:
                return raw_symbol, raw_phase

            last_symbol = run_symbols[-1]
            last_phase = run_phases[-1]
            raw_is_alternating = raw_phase != last_phase
            raw_is_duplicate_reset = raw_symbol == last_symbol
            if raw_is_alternating or raw_is_duplicate_reset:
                return raw_symbol, raw_phase

            best: tuple[tuple[int, int], int, bool] | None = None
            for candidate_symbol, candidate_phase, penalty in candidates[1:]:
                if candidate_phase != last_phase:
                    score = (penalty, 0)
                elif candidate_symbol == last_symbol:
                    score = (penalty, 1)
                else:
                    continue

                if best is None or score < best[0]:
                    best = (score, candidate_symbol, candidate_phase)

            if best is None:
                return raw_symbol, raw_phase
            return best[1], best[2]

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

        def commit_pending_edge() -> None:
            nonlocal pending_edge
            if pending_edge is None:
                return

            x, y, src_phase, dst_phase = pending_edge
            pending_edge = None

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

        while True:
            done = maybe_done()
            if done is not None:
                return done

            packet = yield progress()

            if packet is None:
                pending_edge = None
                run_symbols.clear()
                run_phases.clear()
                continue

            curr_symbol, curr_phase = choose_observed_packet(
                int(bool_array_to_uint16(packet))
            )

            if run_phases and curr_phase == run_phases[-1]:
                is_duplicate_reset = curr_symbol == run_symbols[-1]
                if (not PHASE_REPAIR) or is_duplicate_reset:
                    if pending_edge is not None:
                        if is_duplicate_reset:
                            pending_edge = None
                        else:
                            commit_pending_edge()
                    run_symbols[:] = [curr_symbol]
                    run_phases[:] = [curr_phase]
                    continue

                # A same-phase, different-symbol packet is more often a flipped
                # phase observation than a real delimiter inside guarded streams.
                curr_phase = not curr_phase

            run_symbols.append(curr_symbol)
            run_phases.append(curr_phase)

            if len(run_symbols) > max_run_keep:
                run_symbols[:] = run_symbols[-max_run_keep:]
                run_phases[:] = run_phases[-max_run_keep:]

            if len(run_symbols) < (PREFIX + 1):
                continue

            commit_pending_edge()

            x = pack_prefix_symbols(run_symbols[-(PREFIX + 1) : -1], z)
            y = run_symbols[-1]
            src_phase = run_phases[-2]
            dst_phase = run_phases[-1]

            pending_edge = (x, y, src_phase, dst_phase)

    return Protocol(
        make_sampler=make_sampler,
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
