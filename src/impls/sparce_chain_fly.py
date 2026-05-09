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
SALT = 0x9E3779B97F4A7C15


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
) -> int:
    subset = subset_from_x(
        x, len(coefficients), cdf, salt, singleton_limit=len(coefficients)
    )
    y = 0
    for idx in subset:
        y ^= int(coefficients[idx])
    return y


def _select_agreed_xs(m: int) -> List[int]:
    # With singleton_limit=m, x=0..m-1 gives the identity X matrix.
    return list(range(m))


def _solve_coefficients_from_agreed_y(
    agreed_xs: List[int],
    y_columns: np.ndarray,
    cdf: List[float],
    salt: int,
) -> np.ndarray:
    m = len(agreed_xs)
    symbols: List[int | None] = [None for _ in range(m)]
    pending: List[Tuple[Set[int], int]] = []

    for x, y in zip(agreed_xs, y_columns):
        peel_add_equation(int(x), int(y), m, cdf, salt, symbols, pending)
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
    agreed_xs = _select_agreed_xs(m)

    max_bits = max_message_bitsize(packet_bitsize)
    if message_bitsize > max_bits:
        raise ValueError(
            "message_bitsize too large for sparce_chain_fly: "
            f"max {max_bits} for packet_bitsize={packet_bitsize}"
        )

    def make_sampler(message: Message) -> Sampler:
        y_columns = _stuff_message(message, N, m)
        coefficients = _solve_coefficients_from_agreed_y(
            agreed_xs, y_columns, cdf, SALT
        )

        def f(x: int) -> int:
            return _eval_sparse_coefficients(x, coefficients, cdf, SALT)

        nxt = np.empty(delimiter, dtype=np.uint16)
        for x in range(delimiter):
            nxt[x] = np.uint16(f(x))

        window = m
        recent_counts = np.zeros(delimiter, dtype=np.int32)
        recent: deque[int] = deque()

        probes = min(SEGMENT_PROBES, delimiter)
        rng = np.random.default_rng(_sampler_seed(coefficients, N, message_bitsize))

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
            eligible = np.flatnonzero(recent_counts == 0)
            if eligible.size == 0:
                eligible = np.arange(delimiter, dtype=np.int32)

            probe_count = min(probes, int(eligible.size))
            if probe_count <= 0:
                probe_count = 1

            return rng.choice(eligible, size=probe_count, replace=False)

        delimiter_packet = np.ones(N, dtype=np.bool_)
        while True:
            starts = choose_probe_starts()

            best_segment: List[int] = []
            for start in starts:
                candidate = build_virtual_segment(int(start))
                if len(candidate) > len(best_segment):
                    best_segment = candidate

            if not best_segment:
                best_segment = [0, int(nxt[0])]

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
                peel_add_equation(x, y, m, cdf, SALT, coefficients, pending)
                peel_propagate(coefficients, pending)
            x = y

        coeff_array = np.array(coefficients, dtype=np.uint16)
        y_columns = np.empty(m, dtype=np.uint16)
        for i, agreed_x in enumerate(agreed_xs):
            y_columns[i] = np.uint16(
                _eval_sparse_coefficients(agreed_x, coeff_array, cdf, SALT)
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
