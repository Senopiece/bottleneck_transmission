import math
from collections import deque
from typing import List, Set, Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import (
    bool_array_to_uint16,
    uint16_to_bool_array,
)
from ._utils.sparce import (
    peel_propagate,
    robust_soliton_cdf,
    subset_from_x,
)

# Domain:
# skip_probability: [0, 1)
# corruption_probability: 0
# skip_observation: 1.0

SEGMENT_PROBES = 4
SALT = 0x9E3779B97F4A7C15
HARDCODED_SALTS: dict[tuple[int, int], int] = {
    (5, 2): 0x0000000000000000,
    (5, 3): 0x0000000000000047,
    (5, 4): 0x2189E53415694C18,
    (5, 5): 0x4CDACBB36DEB8C11,
    (5, 6): 0x720BA37C8CC9E713,
    (5, 7): 0x720BA37C8CC9E713,
    (5, 8): 0x50486B04304D489C,
    (5, 9): 0x650C769DB2ADE9E2,
    (5, 10): 0x0F5A9483E6FCCAD2,
    (5, 11): 0x720BA37C8CC9E713,
    (5, 12): 0x7E90960A3C08E584,
    (5, 13): 0x41481E6D02854FDC,
    (5, 14): 0x7E90960A3C08E584,
    (5, 15): 0x41481E6D02854FDC,
    (5, 16): 0x1F36787BCD01BB34,
    (5, 17): 0x7E90960A3C08E584,
    (5, 18): 0x7E90960A3C08E584,
    (5, 19): 0x7E90960A3C08E584,
    (5, 20): 0x650C769DB2ADE9E2,
    (5, 21): 0x6B1C6E4FDD4A765B,
    (5, 22): 0x50486B04304D489C,
    (5, 23): 0x00232D25061485D4,
    (5, 24): 0x22E12324C53FB16A,
    (5, 25): 0x7E90960A3C08E584,
    (5, 26): 0x0000000000000029,
    (5, 27): 0x0FC761580AFF8B2C,
    (5, 28): 0x7E42F9F58A4E58CC,
    (5, 29): 0x7E42F9F58A4E58CC,
    (5, 30): 0x0000000000000059,
}


def _raw_block_count(message_bitsize: int, packet_bitsize: int) -> int:
    if message_bitsize <= 0:
        return 0
    return math.ceil(message_bitsize / packet_bitsize)


def _check_capacity(packet_bitsize: int, raw_blocks: int) -> None:
    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2")

    delimiter = (1 << packet_bitsize) - 1
    stuffed_blocks = raw_blocks + 1

    if not _has_stuffing_capacity(packet_bitsize, raw_blocks):
        raise ValueError(
            "message_bitsize too large for sparce_chain_fly: "
            f"requires m={stuffed_blocks}, but delimiter-safe linked stuffing needs m < {delimiter}"
        )


def _has_stuffing_capacity(packet_bitsize: int, raw_blocks: int) -> bool:
    delimiter = (1 << packet_bitsize) - 1
    stuffed_blocks = raw_blocks + 1
    return stuffed_blocks < delimiter


def _message_to_raw_blocks(
    message: Message, packet_bitsize: int, raw_blocks: int
) -> np.ndarray:
    raw_bits = raw_blocks * packet_bitsize
    padded = np.zeros(raw_bits, dtype=np.bool_)
    if message.shape[0] > 0:
        padded[: message.shape[0]] = message

    blocks = np.zeros(raw_blocks, dtype=np.uint16)
    for i in range(raw_blocks):
        value = 0
        for bit in padded[i * packet_bitsize : (i + 1) * packet_bitsize]:
            value = (value << 1) | int(bit)
        blocks[i] = np.uint16(value)
    return blocks


def _raw_blocks_to_message(
    raw_blocks: np.ndarray, packet_bitsize: int, message_bitsize: int
) -> np.ndarray:
    padded = np.zeros(raw_blocks.shape[0] * packet_bitsize, dtype=np.bool_)
    for i, block in enumerate(raw_blocks):
        value = int(block)
        for j in range(packet_bitsize):
            shift = packet_bitsize - 1 - j
            padded[i * packet_bitsize + j] = bool((value >> shift) & 1)
    return padded[:message_bitsize]


def _stuff_message(message: Message, packet_bitsize: int, m: int) -> np.ndarray:
    delimiter = (1 << packet_bitsize) - 1
    if m >= delimiter:
        raise ValueError("linked stuffing requires m to be below the delimiter value")

    raw = _message_to_raw_blocks(message, packet_bitsize, m - 1)
    stuffed = np.empty(m, dtype=np.uint16)
    stuffed[1:] = raw

    all_ones_positions = [
        pos for pos, block in enumerate(raw, start=1) if int(block) == delimiter
    ]

    if all_ones_positions:
        stuffed[0] = np.uint16(all_ones_positions[0])
    else:
        stuffed[0] = np.uint16(m)

    for idx, pos in enumerate(all_ones_positions):
        next_pos = (
            all_ones_positions[idx + 1] if idx + 1 < len(all_ones_positions) else m
        )
        stuffed[pos] = np.uint16(next_pos)

    if np.any(stuffed == delimiter):
        raise RuntimeError("linked stuffing produced a delimiter column")

    return stuffed


def _unstuff_message(
    stuffed: np.ndarray,
    packet_bitsize: int,
    message_bitsize: int,
) -> np.ndarray:
    delimiter = (1 << packet_bitsize) - 1
    m = stuffed.shape[0]
    if m >= delimiter:
        raise ValueError("linked stuffing requires m to be below the delimiter value")
    if np.any(stuffed == delimiter):
        raise ValueError("stuffed message contains delimiter block")

    raw = np.array(stuffed[1:], dtype=np.uint16, copy=True)
    pointer = int(stuffed[0])
    if pointer < 1 or pointer > m:
        raise ValueError("invalid stuffing head pointer")

    while pointer < m:
        next_pointer = int(stuffed[pointer])
        if next_pointer <= pointer or next_pointer > m:
            raise ValueError("invalid linked stuffing pointer chain")
        raw[pointer - 1] = np.uint16(delimiter)
        pointer = next_pointer

    return _raw_blocks_to_message(raw, packet_bitsize, message_bitsize)


def _eval_sparse_coefficients(
    x: int,
    coefficients: List[int] | np.ndarray,
    cdf: List[float],
    salt: int,
    singleton_by_x: dict[int, int],
) -> int:
    singleton_idx = singleton_by_x.get(int(x))
    if singleton_idx is not None:
        return int(coefficients[singleton_idx])

    subset = subset_from_x(x, len(coefficients), cdf, salt, singleton_limit=0)
    y = 0
    for idx in subset:
        y ^= int(coefficients[idx])
    return y


def _select_agreed_xs(packet_bitsize: int, m: int) -> List[int]:
    delimiter = (1 << packet_bitsize) - 1
    if not (0 <= m < delimiter):
        raise ValueError("cannot select delimiter-safe agreed x values")

    # Keep pointer positions aligned with their stuffed columns, but spend one
    # singleton on m because it is the common byte-stuffing "end" pointer.
    if m == 0:
        return []
    return list(range(m - 1)) + [m]


def _select_salt(packet_bitsize: int, m: int) -> int:
    return HARDCODED_SALTS.get((packet_bitsize, m), SALT)


def _peel_add_equation(
    x: int,
    y: int,
    k: int,
    cdf: List[float],
    salt: int,
    singleton_by_x: dict[int, int],
    symbols: List[int | None],
    pending: List[Tuple[Set[int], int]],
) -> bool:
    if k <= 0:
        return False

    singleton_idx = singleton_by_x.get(int(x))
    if singleton_idx is not None:
        subset = [singleton_idx]
    else:
        subset = subset_from_x(x, k, cdf, salt, singleton_limit=0)

    rhs = int(y)
    unknown: Set[int] = set()
    for idx in subset:
        known = symbols[idx]
        if known is None:
            unknown.add(idx)
        else:
            rhs ^= known

    if not unknown:
        return False

    if len(unknown) == 1:
        idx = next(iter(unknown))
        if symbols[idx] is None:
            symbols[idx] = rhs
            return True
        return False

    pending.append((unknown, rhs))
    return False


def _solve_coefficients_from_agreed_y(
    agreed_xs: List[int],
    y_columns: np.ndarray,
    cdf: List[float],
    salt: int,
    singleton_by_x: dict[int, int],
) -> np.ndarray:
    m = len(agreed_xs)
    symbols: List[int | None] = [None for _ in range(m)]
    pending: List[Tuple[Set[int], int]] = []

    for x, y in zip(agreed_xs, y_columns):
        _peel_add_equation(
            int(x), int(y), m, cdf, salt, singleton_by_x, symbols, pending
        )
        peel_propagate(symbols, pending)

    if any(v is None for v in symbols):
        raise RuntimeError("agreed sparse system is not peel-solvable")

    return np.array(symbols, dtype=np.uint16)


def _sampler_seed(
    coefficients: np.ndarray,
    packet_bitsize: int,
    message_bitsize: int,
) -> int:
    seed = 1469598103934665603
    fnv_prime = 1099511628211

    seed ^= packet_bitsize & 0xFFFF
    seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    seed ^= message_bitsize & 0xFFFFFFFF
    seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF

    for coeff in coefficients:
        seed ^= int(coeff) + 1
        seed = (seed * fnv_prime) & 0xFFFFFFFFFFFFFFFF

    return seed


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = int(config.packet_bitsize)
    message_bitsize = int(config.message_bitsize)

    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    raw_blocks = _raw_block_count(message_bitsize, packet_bitsize)
    _check_capacity(packet_bitsize, raw_blocks)

    N = packet_bitsize
    delimiter = (1 << N) - 1
    m = raw_blocks + 1
    cdf = robust_soliton_cdf(m)
    salt = _select_salt(packet_bitsize, m)
    agreed_xs = _select_agreed_xs(packet_bitsize, m)
    singleton_by_x = {int(x): idx for idx, x in enumerate(agreed_xs)}

    max_bits = max_message_bitsize(packet_bitsize)
    if message_bitsize > max_bits:
        raise ValueError(
            "message_bitsize too large for sparce_chain_fly: "
            f"max {max_bits} for packet_bitsize={packet_bitsize}"
        )

    def make_sampler(message: Message) -> Sampler:
        y_columns = _stuff_message(message, N, m)
        coefficients = _solve_coefficients_from_agreed_y(
            agreed_xs, y_columns, cdf, salt, singleton_by_x
        )

        def f(x: int) -> int:
            return _eval_sparse_coefficients(x, coefficients, cdf, salt, singleton_by_x)

        nxt = np.empty(delimiter, dtype=np.uint16)
        for x in range(delimiter):
            nxt[x] = np.uint16(f(x))

        window = m
        recent_counts = np.zeros(delimiter, dtype=np.int32)
        recent: deque[int] = deque()

        probes = min(SEGMENT_PROBES, delimiter)
        rng = np.random.default_rng(_sampler_seed(coefficients, N, message_bitsize))
        singleton_emit_counts = np.zeros(m, dtype=np.int32)
        agreed_x_array = np.array(agreed_xs, dtype=np.int32)

        def remember(packet_id: int) -> None:
            if window <= 0:
                return
            recent.append(packet_id)
            recent_counts[packet_id] += 1
            if len(recent) > window:
                dropped = recent.popleft()
                recent_counts[dropped] -= 1

        def build_virtual_segment(start: int) -> List[int]:
            visited: Set[int] = {start}
            segment = [start]
            cur = start

            while True:
                nxt_id = int(nxt[cur])
                segment.append(nxt_id)

                if nxt_id == delimiter:
                    return segment
                if recent_counts[nxt_id] > 0 or nxt_id in visited:
                    return segment

                visited.add(nxt_id)
                cur = nxt_id

        def choose_probe_starts() -> np.ndarray:
            min_singleton_count = int(np.min(singleton_emit_counts))
            singleton_starts = agreed_x_array[
                singleton_emit_counts == min_singleton_count
            ]
            eligible_singletons = singleton_starts[recent_counts[singleton_starts] == 0]
            if eligible_singletons.size == 0:
                eligible_singletons = singleton_starts

            if eligible_singletons.size > 0:
                probe_count = min(probes, int(eligible_singletons.size))
                return rng.choice(
                    eligible_singletons.astype(np.int32),
                    size=probe_count,
                    replace=False,
                )

            eligible = np.flatnonzero(recent_counts == 0)
            if eligible.size == 0:
                eligible = np.arange(delimiter, dtype=np.int32)

            probe_count = min(probes, int(eligible.size))
            if probe_count <= 0:
                probe_count = 1

            return rng.choice(eligible, size=probe_count, replace=False)

        def segment_score(segment: List[int]) -> Tuple[int, float, int]:
            rare_edges = 0
            weighted_freshness = 0.0
            min_count = int(np.min(singleton_emit_counts))

            for curr_state, next_state in zip(segment, segment[1:]):
                if next_state == delimiter:
                    continue
                singleton_idx = singleton_by_x.get(int(curr_state))
                if singleton_idx is None:
                    continue
                count = int(singleton_emit_counts[singleton_idx])
                if count == min_count:
                    rare_edges += 1
                weighted_freshness += 1.0 / float(count + 1)

            return (rare_edges, weighted_freshness, len(segment))

        delimiter_packet = np.ones(N, dtype=np.bool_)
        while True:
            starts = choose_probe_starts()

            best_segment: List[int] = []
            best_score: Tuple[int, float, int] | None = None
            for start in starts:
                candidate = build_virtual_segment(int(start))
                candidate_score = segment_score(candidate)
                if best_score is None or candidate_score > best_score:
                    best_segment = candidate
                    best_score = candidate_score

            if not best_segment:
                best_segment = [0, int(nxt[0])]

            for curr_state, next_state in zip(best_segment, best_segment[1:]):
                singleton_idx = singleton_by_x.get(int(curr_state))
                if singleton_idx is not None and next_state != delimiter:
                    singleton_emit_counts[singleton_idx] += 1

            for state in best_segment:
                if state == delimiter:
                    yield delimiter_packet
                else:
                    remember(state)
                    yield uint16_to_bool_array(np.uint16(state), N)

            yield delimiter_packet

    def make_estimator() -> Estimator:
        coefficients: List[int | None] = [None for _ in range(m)]
        pending: List[Tuple[Set[int], int]] = []
        seen_x: Set[int] = set()
        x: int | None = None

        def progress() -> float:
            return sum(1 for v in coefficients if v is not None) / m

        while any(v is None for v in coefficients):
            packet = yield progress()

            if packet is None or np.all(packet == 1):
                x = None
                continue

            y = int(bool_array_to_uint16(packet))
            if x is not None and x not in seen_x:
                seen_x.add(x)
                _peel_add_equation(
                    x, y, m, cdf, salt, singleton_by_x, coefficients, pending
                )
                peel_propagate(coefficients, pending)
            x = y

        coeff_array = np.array(coefficients, dtype=np.uint16)
        y_columns = np.empty(m, dtype=np.uint16)
        for i, agreed_x in enumerate(agreed_xs):
            y_columns[i] = np.uint16(
                _eval_sparse_coefficients(
                    agreed_x, coeff_array, cdf, salt, singleton_by_x
                )
            )

        return _unstuff_message(y_columns, N, message_bitsize)

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if packet_bitsize <= 1:
        return 0

    delimiter = (1 << packet_bitsize) - 1
    return packet_bitsize * max(0, delimiter - 2)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
