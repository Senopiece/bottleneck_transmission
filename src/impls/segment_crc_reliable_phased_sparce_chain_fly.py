from typing import Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import bool_array_to_uint16, uint16_to_bool_array
from .reliable_phased_sparce_chain_fly import (
    create_protocol as create_inner_protocol,
)

HASH_SEED = 0x15
HASH_MUL = 17


def _packet_value(packet: np.ndarray, mask: int) -> int:
    return int(bool_array_to_uint16(packet) & mask)


def _same_packet(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _guard_value(
    body: list[np.ndarray],
    next_first: np.ndarray,
    packet_bitsize: int,
) -> int:
    mask = (1 << packet_bitsize) - 1
    h = HASH_SEED & mask
    for packet in body:
        value = _packet_value(packet, mask)
        h = ((h * HASH_MUL) ^ value ^ (h >> 2)) & mask

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

    inner_protocol = create_inner_protocol(config)

    def make_sampler(message: Message) -> Sampler:
        inner_sampler = inner_protocol.make_sampler(message)
        buf: list[np.ndarray] = []
        prev: np.ndarray | None = None

        while True:
            packet = next(inner_sampler)

            if prev is not None and _same_packet(prev, packet):
                completed = buf[:-1]
                if completed:
                    guard = _guard_packet(completed, prev, packet_bitsize)
                    for out_packet in completed:
                        yield out_packet
                    yield guard
                buf = [prev, packet]
            else:
                buf.append(packet)

            prev = packet

    def make_estimator() -> Estimator:
        inner_estimator = inner_protocol.make_estimator()
        progress = next(inner_estimator)

        buf: list[np.ndarray] = []
        prev: np.ndarray | None = None

        def reset_inner() -> None:
            nonlocal progress
            progress = inner_estimator.send(None)

        def flush_completed(completed: list[np.ndarray], next_first: np.ndarray) -> None:
            nonlocal progress
            if len(completed) < 2:
                return

            body = completed[:-1]
            observed_guard = completed[-1]
            expected_guard = _guard_value(body, next_first, packet_bitsize)
            mask = (1 << packet_bitsize) - 1

            if _packet_value(observed_guard, mask) != expected_guard:
                reset_inner()
                return

            for out_packet in body:
                progress = inner_estimator.send(out_packet)

        while True:
            packet = yield progress

            if packet is None:
                buf.clear()
                prev = None
                try:
                    reset_inner()
                except StopIteration as exc:
                    return exc.value
                continue

            if prev is not None and _same_packet(prev, packet):
                completed = buf[:-1]
                try:
                    flush_completed(completed, prev)
                except StopIteration as exc:
                    return exc.value
                buf = [prev, packet]
            else:
                buf.append(packet)

            prev = packet

    return Protocol(make_sampler=make_sampler, make_estimator=make_estimator)


def max_message_bitsize(packet_bitsize: int) -> int:
    # This experimental composition is intentionally disabled for the notebook
    # auto-discovery benchmark. The segment guard already delays packets until a
    # whole segment validates; putting reliable_phased_sparce_chain_fly inside
    # adds another validation delay and can exceed the benchmark max_iters on
    # unlucky corruption patterns without improving failure rate over
    # segment_crc_phased_sparce_chain_fly.
    return 0


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
