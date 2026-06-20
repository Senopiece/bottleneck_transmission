import math
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import make_message_vector, message_from_message_vector
from .phased_sparce_chain_fly import create_protocol as create_base_protocol

# Outer systematic XOR checks over message symbols.
# Packet format is unchanged; packet_bitsize is unchanged. We spend a small
# number of extra message symbols, then let the regular phased sparse-chain
# protocol transmit the protected message.

CHECK_RATE = 0.08
MIN_CHECK_SYMBOLS = 4
CHECK_REPETITIONS = 3
REPAIR_PASSES = 3
VOTE_THRESHOLD = 2
SALT = 0xA24BAED4963EE407


def _symbol_bits(packet_bitsize: int) -> int:
    if packet_bitsize <= 1:
        return 0
    return packet_bitsize - 1


def _data_symbol_count(message_bitsize: int, z: int) -> int:
    if message_bitsize <= 0:
        return 0
    return math.ceil(message_bitsize / z)


def _check_symbol_count(data_symbols: int) -> int:
    if data_symbols <= 0:
        return 0
    return max(MIN_CHECK_SYMBOLS, math.ceil(data_symbols * CHECK_RATE))


def _check_subsets(data_symbols: int, check_symbols: int) -> List[List[int]]:
    subsets: List[List[int]] = [[] for _ in range(check_symbols)]
    if data_symbols <= 0 or check_symbols <= 0:
        return subsets

    for idx in range(data_symbols):
        rng = random.Random((SALT ^ (idx * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF)
        chosen: set[int] = set()
        attempts = 0
        while len(chosen) < min(CHECK_REPETITIONS, check_symbols):
            chosen.add(rng.randrange(check_symbols))
            attempts += 1
            if attempts > 16 * check_symbols:
                break
        for check_idx in chosen:
            subsets[check_idx].append(idx)

    return subsets


def _encode_symbols(data: np.ndarray, q: int, check_subsets: List[List[int]]) -> np.ndarray:
    check_values = np.zeros(len(check_subsets), dtype=np.uint16)
    mask = q - 1
    for check_idx, subset in enumerate(check_subsets):
        value = 0
        for idx in subset:
            value ^= int(data[idx])
        check_values[check_idx] = np.uint16(value & mask)
    return np.concatenate((data.astype(np.uint16, copy=False), check_values))


def _repair_symbols(
    encoded: np.ndarray,
    data_symbols: int,
    q: int,
    check_subsets: List[List[int]],
) -> np.ndarray:
    if data_symbols <= 0:
        return np.empty(0, dtype=np.uint16)

    data = np.array(encoded[:data_symbols], dtype=np.uint16, copy=True)
    checks = np.array(encoded[data_symbols:], dtype=np.uint16, copy=False)
    mask = q - 1

    for _ in range(REPAIR_PASSES):
        votes: List[dict[int, int]] = [defaultdict(int) for _ in range(data_symbols)]

        for check_idx, subset in enumerate(check_subsets):
            expected = 0
            for idx in subset:
                expected ^= int(data[idx])
            residual = (expected ^ int(checks[check_idx])) & mask
            if residual == 0:
                continue

            for idx in subset:
                votes[idx][int(data[idx]) ^ residual] += 1

        changed = False
        for idx, vote_map in enumerate(votes):
            if not vote_map:
                continue
            ranked = sorted(vote_map.items(), key=lambda item: item[1], reverse=True)
            best_value, best_count = ranked[0]
            second_count = ranked[1][1] if len(ranked) > 1 else 0
            if best_count >= VOTE_THRESHOLD and best_count > second_count:
                if int(data[idx]) != best_value:
                    data[idx] = np.uint16(best_value & mask)
                    changed = True

        if not changed:
            break

    return data


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = int(config.packet_bitsize)
    message_bitsize = int(config.message_bitsize)

    z = _symbol_bits(packet_bitsize)
    if z <= 0:
        raise ValueError("packet_bitsize must be >= 2")
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    q = 1 << z
    data_symbols = _data_symbol_count(message_bitsize, z)
    check_symbols = _check_symbol_count(data_symbols)
    check_subsets = _check_subsets(data_symbols, check_symbols)
    encoded_symbols = data_symbols + check_symbols
    encoded_bitsize = encoded_symbols * z

    base_protocol = create_base_protocol(
        Config(packet_bitsize=packet_bitsize, message_bitsize=encoded_bitsize)
    )

    def make_sampler(message: Message) -> Sampler:
        data = make_message_vector(message, data_symbols, q)
        encoded = _encode_symbols(data, q, check_subsets)
        encoded_bits = message_from_message_vector(encoded, encoded_bitsize, q)
        return base_protocol.make_sampler(encoded_bits)

    def make_estimator() -> Estimator:
        base_estimator = base_protocol.make_estimator()
        progress = next(base_estimator)

        while True:
            packet = yield progress
            try:
                progress = base_estimator.send(packet)
            except StopIteration as exc:
                encoded_bits = exc.value
                encoded = make_message_vector(encoded_bits, encoded_symbols, q)
                repaired = _repair_symbols(encoded, data_symbols, q, check_subsets)
                return message_from_message_vector(repaired, message_bitsize, q)

    return Protocol(make_sampler=make_sampler, make_estimator=make_estimator)


def max_message_bitsize(packet_bitsize: int) -> int:
    z = _symbol_bits(packet_bitsize)
    if z <= 0:
        return 0

    base_max = create_base_max_message_bitsize(packet_bitsize)
    if base_max <= 0:
        return 0

    # Search is tiny and avoids algebraic off-by-one errors from ceil/check rate.
    best = 0
    for bits in range(base_max + 1):
        data_symbols = _data_symbol_count(bits, z)
        check_symbols = _check_symbol_count(data_symbols)
        if (data_symbols + check_symbols) * z <= base_max:
            best = bits
    return best


def create_base_max_message_bitsize(packet_bitsize: int) -> int:
    from .phased_sparce_chain_fly import max_message_bitsize

    return max_message_bitsize(packet_bitsize)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
