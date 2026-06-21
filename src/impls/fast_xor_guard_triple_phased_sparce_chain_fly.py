from typing import Tuple

import numpy as np

from . import phased_sparce_chain_fly as base_module
from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils import sparce

# Same 3+1 xor guard as xor_guard_triple_phased_sparce_chain_fly, but with a
# more aggressive robust-soliton distribution inside the sparse chain.
# It also gives segment length a small sender-side tie-break bonus. It keeps
# PREFIX=2, so decode complexity stays in the same small state space.

GROUP_SIZE = 3
SOLITON_C = 0.02
SOLITON_DELTA = 0.65
SEGMENT_LENGTH_BONUS = 0.0
MIN_TUNED_MESSAGE_BITSIZE = 96


def _xor_packets(packets: list[np.ndarray]) -> np.ndarray:
    result = np.array(packets[0], dtype=np.bool_, copy=True)
    for packet in packets[1:]:
        result = np.logical_xor(result, packet)
    return result


def _same_packet(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _create_base_protocol(config: Config) -> Protocol:
    if config.message_bitsize < MIN_TUNED_MESSAGE_BITSIZE:
        return base_module.create_protocol(config)

    old_robust_soliton_cdf = base_module.robust_soliton_cdf
    old_segment_length_bonus = base_module.SEGMENT_LENGTH_BONUS

    def tuned_robust_soliton_cdf(k: int) -> list[float]:
        return sparce.robust_soliton_cdf(k, c=SOLITON_C, delta=SOLITON_DELTA)

    base_module.robust_soliton_cdf = tuned_robust_soliton_cdf
    base_module.SEGMENT_LENGTH_BONUS = SEGMENT_LENGTH_BONUS
    try:
        return base_module.create_protocol(config)
    finally:
        base_module.robust_soliton_cdf = old_robust_soliton_cdf
        base_module.SEGMENT_LENGTH_BONUS = old_segment_length_bonus


def create_protocol(config: Config) -> Protocol:
    base_protocol = _create_base_protocol(config)

    def make_sampler(message: Message) -> Sampler:
        base_sampler = base_protocol.make_sampler(message)
        group: list[np.ndarray] = []

        while True:
            group.append(np.array(next(base_sampler), dtype=np.bool_, copy=True))
            if len(group) < GROUP_SIZE:
                continue

            for packet in group:
                yield packet
            yield _xor_packets(group)
            group.clear()

    def make_estimator() -> Estimator:
        base_estimator = base_protocol.make_estimator()
        progress = next(base_estimator)
        window: list[np.ndarray] = []

        def reset_inner() -> None:
            nonlocal progress
            progress = base_estimator.send(None)

        def feed(packet: np.ndarray) -> None:
            nonlocal progress
            progress = base_estimator.send(packet)

        while True:
            packet = yield progress

            if packet is None:
                window.clear()
                try:
                    reset_inner()
                except StopIteration as exc:
                    return exc.value
                continue

            window.append(np.array(packet, dtype=np.bool_, copy=True))

            while len(window) >= GROUP_SIZE + 1:
                data = window[:GROUP_SIZE]
                guard = window[GROUP_SIZE]
                if _same_packet(_xor_packets(data), guard):
                    try:
                        for data_packet in data:
                            feed(data_packet)
                    except StopIteration as exc:
                        return exc.value
                    del window[: GROUP_SIZE + 1]
                else:
                    try:
                        reset_inner()
                    except StopIteration as exc:
                        return exc.value
                    del window[0]

    return Protocol(make_sampler=make_sampler, make_estimator=make_estimator)


def max_message_bitsize(packet_bitsize: int) -> int:
    return base_module.max_message_bitsize(packet_bitsize)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
