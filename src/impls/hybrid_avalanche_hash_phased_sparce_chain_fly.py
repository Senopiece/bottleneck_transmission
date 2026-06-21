from typing import Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import bool_array_to_uint16, uint16_to_bool_array
from .phased_sparce_chain_fly import (
    create_protocol as create_base_protocol,
    max_message_bitsize as base_max_message_bitsize,
)

# Hybrid segment protection with stronger tiny hashes:
# - every segment gets a free in-band 4-bit hash in the existing boundary marker;
# - long segments additionally get one full guard packet.
#
# The packet format and packet count are identical to hybrid_hash_crc; this only
# replaces the small linear mixers with a 64-bit avalanche reduced to 4/5 bits.

HASH_SEED = 0x0B
CRC_HASH_SEED = 0x15
MIN_STRONG_GUARD_PACKETS = 2

_MIX_CONST = 0x9E3779B97F4A7C15


def _mix64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return x & 0xFFFFFFFFFFFFFFFF


def _same_packet(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _packet_value(packet: np.ndarray, mask: int) -> int:
    return int(bool_array_to_uint16(packet) & mask)


def _segment_hash(segment: list[np.ndarray], packet_bitsize: int) -> int:
    z = packet_bitsize - 1
    mask = (1 << z) - 1
    full_mask = (1 << packet_bitsize) - 1
    h = _mix64(HASH_SEED ^ (packet_bitsize << 16) ^ len(segment))
    for i, packet in enumerate(segment):
        value = _packet_value(packet, full_mask)
        h = _mix64(h ^ _mix64(value + _MIX_CONST * (i + 1)))
    return h & mask


def _hash_marker(phase_b: bool, hash_value: int, packet_bitsize: int) -> np.ndarray:
    z = packet_bitsize - 1
    value = int(hash_value) | ((1 << z) if phase_b else 0)
    return uint16_to_bool_array(np.uint16(value), packet_bitsize)


def _guard_value(
    body: list[np.ndarray],
    next_first: np.ndarray,
    packet_bitsize: int,
) -> int:
    mask = (1 << packet_bitsize) - 1
    h = _mix64(
        CRC_HASH_SEED
        ^ (packet_bitsize << 17)
        ^ (len(body) << 3)
        ^ _packet_value(next_first, mask)
    )
    for i, packet in enumerate(body):
        value = _packet_value(packet, mask)
        h = _mix64(h ^ _mix64(value + _MIX_CONST * (i + 5)))
    h &= mask

    forbidden = {_packet_value(next_first, mask)}
    if body:
        forbidden.add(_packet_value(body[-1], mask))

    while h in forbidden:
        h = (h + 1) & mask
    return h


def _guard_packet(
    body: list[np.ndarray],
    next_first: np.ndarray,
    packet_bitsize: int,
) -> np.ndarray:
    return uint16_to_bool_array(
        np.uint16(_guard_value(body, next_first, packet_bitsize)), packet_bitsize
    )


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = int(config.packet_bitsize)
    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2")

    z = packet_bitsize - 1
    symbol_mask = (1 << z) - 1
    full_mask = (1 << packet_bitsize) - 1
    base_protocol = create_base_protocol(config)

    def make_sampler(message: Message) -> Sampler:
        base_sampler = base_protocol.make_sampler(message)
        buf: list[np.ndarray] = []
        prev: np.ndarray | None = None

        while True:
            packet = next(base_sampler)

            if prev is not None and _same_packet(prev, packet):
                completed = buf[:-1]
                if completed:
                    marker = _hash_marker(
                        bool(packet[0]),
                        _segment_hash(completed, packet_bitsize),
                        packet_bitsize,
                    )
                    yield marker
                    if len(completed) >= MIN_STRONG_GUARD_PACKETS:
                        yield _guard_packet(completed, prev, packet_bitsize)
                else:
                    yield packet
                buf = [prev, packet]
            else:
                yield packet
                buf.append(packet)

            prev = packet

    def make_estimator() -> Estimator:
        base_estimator = base_protocol.make_estimator()
        progress = next(base_estimator)

        buf: list[np.ndarray] = []
        stream_prev: np.ndarray | None = None
        awaiting_guard: Tuple[list[np.ndarray], np.ndarray, bool] | None = None

        def reset_inner() -> None:
            nonlocal progress
            progress = base_estimator.send(None)

        def feed_segment(segment: list[np.ndarray]) -> None:
            nonlocal progress
            for out_packet in segment:
                progress = base_estimator.send(out_packet)

        while True:
            packet = yield progress

            if packet is None:
                buf.clear()
                stream_prev = None
                awaiting_guard = None
                try:
                    reset_inner()
                except StopIteration as exc:
                    return exc.value
                continue

            if awaiting_guard is not None:
                completed, next_first, inband_ok = awaiting_guard
                awaiting_guard = None
                observed_guard = _packet_value(packet, full_mask)
                expected_guard = _guard_value(completed, next_first, packet_bitsize)
                try:
                    if inband_ok and observed_guard == expected_guard:
                        feed_segment(completed)
                    else:
                        reset_inner()
                except StopIteration as exc:
                    return exc.value

                first = np.array(next_first, dtype=np.bool_, copy=True)
                buf = [first, np.array(first, dtype=np.bool_, copy=True)]
                stream_prev = first
                continue

            if stream_prev is not None and bool(packet[0]) == bool(stream_prev[0]):
                completed = buf[:-1]
                if completed:
                    observed_hash = int(bool_array_to_uint16(packet) & symbol_mask)
                    expected_hash = _segment_hash(completed, packet_bitsize)
                    inband_ok = observed_hash == expected_hash
                    next_first = np.array(stream_prev, dtype=np.bool_, copy=True)

                    if len(completed) >= MIN_STRONG_GUARD_PACKETS:
                        awaiting_guard = (completed, next_first, inband_ok)
                        continue

                    try:
                        if inband_ok:
                            feed_segment(completed)
                        else:
                            reset_inner()
                    except StopIteration as exc:
                        return exc.value

                first = np.array(stream_prev, dtype=np.bool_, copy=True)
                buf = [first, np.array(first, dtype=np.bool_, copy=True)]
            else:
                buf.append(packet)

            stream_prev = packet

    return Protocol(make_sampler=make_sampler, make_estimator=make_estimator)


def max_message_bitsize(packet_bitsize: int) -> int:
    return base_max_message_bitsize(packet_bitsize)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
