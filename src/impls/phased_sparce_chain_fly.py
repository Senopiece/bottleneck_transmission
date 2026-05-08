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

# Number of previous (n-1)-bit symbols packed into x for f(x).
PREFIX = 1


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
    packet_q = 1 << N

    input_bits = PREFIX * z
    if input_bits > MAX_PREFIX_INPUT_BITS:
        raise ValueError(
            "phased_sparce_chain_fly is tailored for small packet sizes: "
            f"PREFIX*(packet_bitsize-1) must be <= {MAX_PREFIX_INPUT_BITS}, got {input_bits}"
        )
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

        def packet_id(symbol: int, phase_b: bool) -> int:
            return int(symbol) + (symbol_q if phase_b else 0)

        def packet_from_id(value: int) -> tuple[int, bool]:
            return value & symbol_mask, bool(value & symbol_q)

        def phased_output(symbol: int, is_phase_b: bool):
            return uint16_to_bool_array(np.uint16(packet_id(symbol, is_phase_b)), N)

        nxt = np.empty(packet_q, dtype=np.int32)
        for pid in range(packet_q):
            symbol, phase_b = packet_from_id(pid)
            next_symbol = f2(symbol) if phase_b else f1(symbol)
            nxt[pid] = packet_id(next_symbol, not phase_b)

        max_window = max(0, symbol_q - 6)
        window_a = min(max_window, k1)
        window_b = min(max_window, k2)
        recent_counts_a = np.zeros(packet_q, dtype=np.int32)
        recent_counts_b = np.zeros(packet_q, dtype=np.int32)
        recent_a: deque[int] = deque()
        recent_b: deque[int] = deque()

        probes = min(SEGMENT_PROBES, packet_q)
        rng = np.random.default_rng(
            sampler_seed(symbols_a, symbols_b, packet_bitsize, message_bitsize)
        )
        started_singletons = np.zeros(packet_q, dtype=np.bool_)

        def counts_for_phase(phase_b: bool) -> np.ndarray:
            return recent_counts_b if phase_b else recent_counts_a

        def recent_for_phase(phase_b: bool) -> deque[int]:
            return recent_b if phase_b else recent_a

        def window_for_phase(phase_b: bool) -> int:
            return window_b if phase_b else window_a

        def remember(pid: int) -> None:
            _, phase_b = packet_from_id(pid)
            window = window_for_phase(phase_b)
            if window <= 0:
                return
            recent_counts = counts_for_phase(phase_b)
            recent = recent_for_phase(phase_b)
            recent.append(pid)
            recent_counts[pid] += 1
            if len(recent) > window:
                dropped = recent.popleft()
                recent_counts[dropped] -= 1

        def build_virtual_segment(start: int) -> list[int]:
            visited = {start}
            segment = [start]
            cur = start

            while True:
                next_id = int(nxt[cur])
                segment.append(next_id)
                _, next_phase_b = packet_from_id(next_id)
                if counts_for_phase(next_phase_b)[next_id] > 0 or next_id in visited:
                    return segment
                visited.add(next_id)
                cur = next_id

        def choose_probe_starts(start_phase_b: bool) -> np.ndarray:
            phase_offset = symbol_q if start_phase_b else 0
            singleton_count = k2 if start_phase_b else k1
            if singleton_count > 0:
                singleton_ids = np.arange(
                    phase_offset,
                    phase_offset + min(singleton_count, symbol_q),
                    dtype=np.int32,
                )
                pending_singletons = singleton_ids[~started_singletons[singleton_ids]]
                if pending_singletons.size > 0:
                    probe_count = min(probes, int(pending_singletons.size))
                    return rng.choice(
                        pending_singletons, size=probe_count, replace=False
                    )

            phase_ids = np.arange(phase_offset, phase_offset + symbol_q, dtype=np.int32)
            recent_counts = counts_for_phase(start_phase_b)
            eligible = phase_ids[recent_counts[phase_ids] == 0]
            if eligible.size == 0:
                eligible = phase_ids

            probe_count = min(probes, int(eligible.size))
            if probe_count <= 0:
                probe_count = 1

            return rng.choice(eligible, size=probe_count, replace=False)

        def has_pending_singletons(phase_b: bool) -> bool:
            phase_offset = symbol_q if phase_b else 0
            singleton_count = k2 if phase_b else k1
            if singleton_count <= 0:
                return False
            singleton_ids = np.arange(
                phase_offset,
                phase_offset + min(singleton_count, symbol_q),
                dtype=np.int32,
            )
            return bool(np.any(~started_singletons[singleton_ids]))

        def segment_end_phase(segment: list[int]) -> bool:
            _, phase_b = packet_from_id(segment[-1])
            return phase_b

        next_start_phase_b = False
        while True:
            starts = choose_probe_starts(next_start_phase_b)

            best_segment: list[int] = []
            for start in starts:
                candidate = build_virtual_segment(int(start))
                if len(candidate) > len(best_segment):
                    best_segment = candidate

            if not best_segment:
                start_id = packet_id(0, next_start_phase_b)
                best_segment = [start_id, int(nxt[start_id])]

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

            first_symbol, first_phase_b = packet_from_id(best_segment[0])
            yield phased_output(first_symbol, first_phase_b)

            if len(best_segment) > 1:
                started_singletons[best_segment[0]] = True
            for pid in best_segment:
                remember(pid)
                symbol, phase_b = packet_from_id(pid)
                yield phased_output(symbol, phase_b)

            _, next_start_phase_b = packet_from_id(best_segment[-1])

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

        while True:
            done = maybe_done()
            if done is not None:
                return done

            packet = yield progress()

            if packet is None:
                run_symbols.clear()
                run_phases.clear()
                continue

            curr_phase = bool(packet[0])
            curr_symbol = int(bool_array_to_uint16(packet) & symbol_mask)

            if run_phases and curr_phase == run_phases[-1]:
                run_symbols[:] = [curr_symbol]
                run_phases[:] = [curr_phase]
                continue

            run_symbols.append(curr_symbol)
            run_phases.append(curr_phase)

            if len(run_symbols) > max_run_keep:
                run_symbols[:] = run_symbols[-max_run_keep:]
                run_phases[:] = run_phases[-max_run_keep:]

            if len(run_symbols) < (PREFIX + 1):
                continue

            x = pack_prefix_symbols(run_symbols[-(PREFIX + 1) : -1], z)
            y = run_symbols[-1]
            src_phase = run_phases[-2]
            dst_phase = run_phases[-1]

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
