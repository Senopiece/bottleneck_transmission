from typing import Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._confirmed_x_phased_sparce_chain_fly import create_protocol as create_inner_protocol
from .phased_sparce_chain_fly import max_message_bitsize as base_max_message_bitsize

# The confirmed-x receiver rejects non-singleton equations that were seen only
# once. The original phased sender tries to avoid revisiting recent x values, so
# confirmations arrive late. This wrapper repeats each chain segment once,
# providing the receiver with a local second observation without changing the
# packet format or adding per-packet checks.

REPLAYS_PER_SEGMENT = 2


def _same_packet(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def create_protocol(config: Config) -> Protocol:
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
                    for _ in range(REPLAYS_PER_SEGMENT):
                        for out_packet in completed:
                            yield out_packet
                buf = [prev, packet]
            else:
                buf.append(packet)

            prev = packet

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=inner_protocol.make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    return base_max_message_bitsize(packet_bitsize)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
