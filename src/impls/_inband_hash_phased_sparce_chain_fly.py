from typing import Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import bool_array_to_uint16, uint16_to_bool_array
from .phased_sparce_chain_fly import (
    create_protocol as create_base_protocol,
    max_message_bitsize as base_max_message_bitsize,
)

# Segment corruption detector with no extra packets.
#
# phased_sparce_chain_fly already emits a same-packet duplicate at each segment
# boundary to reset the receiver. This wrapper replaces the second boundary
# packet's symbol bits with a small hash of the previous segment while keeping
# the phase bit unchanged. The receiver still sees a same-phase boundary, but it
# can now discard corrupted segments before they poison peeling.

HASH_SEED = 0x0B
HASH_MUL = 13


def _same_packet(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _packet_value(packet: np.ndarray, mask: int) -> int:
    return int(bool_array_to_uint16(packet) & mask)


def _segment_hash(segment: list[np.ndarray], packet_bitsize: int) -> int:
    z = packet_bitsize - 1
    mask = (1 << z) - 1
    h = HASH_SEED & mask
    full_mask = (1 << packet_bitsize) - 1
    for packet in segment:
        value = _packet_value(packet, full_mask)
        h = ((h * HASH_MUL) ^ value ^ (value >> 2) ^ (h >> 1)) & mask
    return h


def _hash_marker(
    phase_b: bool,
    hash_value: int,
    packet_bitsize: int,
) -> np.ndarray:
    z = packet_bitsize - 1
    value = int(hash_value) | ((1 << z) if phase_b else 0)
    return uint16_to_bool_array(np.uint16(value), packet_bitsize)


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = int(config.packet_bitsize)
    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2")

    z = packet_bitsize - 1
    symbol_mask = (1 << z) - 1
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

        # Reconstructed base-protocol segment buffer. Boundary hash packets are
        # not stored here; after each boundary we synthesize the original
        # duplicate marker for the next segment.
        buf: list[np.ndarray] = []
        stream_prev: np.ndarray | None = None

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
                try:
                    reset_inner()
                except StopIteration as exc:
                    return exc.value
                continue

            if stream_prev is not None and bool(packet[0]) == bool(stream_prev[0]):
                completed = buf[:-1]
                if completed:
                    observed_hash = int(bool_array_to_uint16(packet) & symbol_mask)
                    expected_hash = _segment_hash(completed, packet_bitsize)
                    try:
                        if observed_hash == expected_hash:
                            feed_segment(completed)
                        else:
                            reset_inner()
                    except StopIteration as exc:
                        return exc.value

                # Start reconstructing the next base segment. The transformed
                # stream has next_first + hash_marker; the base stream had
                # next_first + next_first.
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
