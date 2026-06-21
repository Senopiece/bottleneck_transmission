from typing import Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from .phased_sparce_chain_fly import (
    create_protocol as create_base_protocol,
    max_message_bitsize as base_max_message_bitsize,
)

# Same packet-pairing layer as paired_phased_sparce_chain_fly, but an observed
# pair mismatch is also forwarded as Deletion to the inner phased decoder. This
# tests whether explicit run breaks are better than silently dropping the bad
# packet and sliding to the next copy.


def _same_packet(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def create_protocol(config: Config) -> Protocol:
    base_protocol = create_base_protocol(config)

    def make_sampler(message: Message) -> Sampler:
        base_sampler = base_protocol.make_sampler(message)
        while True:
            packet = next(base_sampler)
            yield packet
            yield packet

    def make_estimator() -> Estimator:
        base_estimator = base_protocol.make_estimator()
        progress = next(base_estimator)
        pending: np.ndarray | None = None

        def reset_inner() -> None:
            nonlocal progress
            progress = base_estimator.send(None)

        while True:
            packet = yield progress

            if packet is None:
                pending = None
                try:
                    reset_inner()
                except StopIteration as exc:
                    return exc.value
                continue

            curr = np.array(packet, dtype=np.bool_, copy=True)
            if pending is None:
                pending = curr
                continue

            if _same_packet(pending, curr):
                pending = None
                try:
                    progress = base_estimator.send(curr)
                except StopIteration as exc:
                    return exc.value
            else:
                pending = curr
                try:
                    reset_inner()
                except StopIteration as exc:
                    return exc.value

    return Protocol(make_sampler=make_sampler, make_estimator=make_estimator)


def max_message_bitsize(packet_bitsize: int) -> int:
    return base_max_message_bitsize(packet_bitsize)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
