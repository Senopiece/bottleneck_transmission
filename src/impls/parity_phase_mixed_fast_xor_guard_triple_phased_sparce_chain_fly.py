from typing import Tuple

import numpy as np

from . import phased_sparce_chain_fly as base_module
from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils import sparce

# Fast profile with phase mixed into the two low symbol bits:
# physical_phase = logical_phase xor parity(symbol & 0b0011).
# This keeps packet size and guard format unchanged.

GROUP_SIZE = 3
SOLITON_C = 0.02
SOLITON_DELTA = 0.65
SEGMENT_LENGTH_BONUS = 0.0
SEGMENT_PROBES = 2
RECENT_WINDOW_MARGIN = 8
PHASE_MIX_MODE = "parity"
PHASE_MIX_MASK = 0b0011


def _xor_packets(packets: list[np.ndarray]) -> np.ndarray:
    result = np.array(packets[0], dtype=np.bool_, copy=True)
    for packet in packets[1:]:
        result = np.logical_xor(result, packet)
    return result


def _same_packet(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _create_base_protocol(config: Config) -> Protocol:
    old_robust_soliton_cdf = base_module.robust_soliton_cdf
    old_segment_length_bonus = base_module.SEGMENT_LENGTH_BONUS
    old_segment_probes = base_module.SEGMENT_PROBES
    old_recent_window_margin = base_module.RECENT_WINDOW_MARGIN
    old_phase_mixing = base_module.PHASE_MIXING
    old_phase_mix_mode = base_module.PHASE_MIX_MODE
    old_phase_mix_mask = base_module.PHASE_MIX_MASK
    old_phase_symbol_xor_mask = base_module.PHASE_SYMBOL_XOR_MASK
    old_phase_repair = base_module.PHASE_REPAIR
    old_phase_candidate_decoding = base_module.PHASE_CANDIDATE_DECODING

    def tuned_robust_soliton_cdf(k: int) -> list[float]:
        return sparce.robust_soliton_cdf(k, c=SOLITON_C, delta=SOLITON_DELTA)

    base_module.robust_soliton_cdf = tuned_robust_soliton_cdf
    base_module.SEGMENT_LENGTH_BONUS = SEGMENT_LENGTH_BONUS
    base_module.SEGMENT_PROBES = SEGMENT_PROBES
    base_module.RECENT_WINDOW_MARGIN = RECENT_WINDOW_MARGIN
    base_module.PHASE_MIXING = True
    base_module.PHASE_MIX_MODE = PHASE_MIX_MODE
    base_module.PHASE_MIX_MASK = PHASE_MIX_MASK
    base_module.PHASE_SYMBOL_XOR_MASK = 0
    base_module.PHASE_REPAIR = False
    base_module.PHASE_CANDIDATE_DECODING = False
    try:
        return base_module.create_protocol(config)
    finally:
        base_module.robust_soliton_cdf = old_robust_soliton_cdf
        base_module.SEGMENT_LENGTH_BONUS = old_segment_length_bonus
        base_module.SEGMENT_PROBES = old_segment_probes
        base_module.RECENT_WINDOW_MARGIN = old_recent_window_margin
        base_module.PHASE_MIXING = old_phase_mixing
        base_module.PHASE_MIX_MODE = old_phase_mix_mode
        base_module.PHASE_MIX_MASK = old_phase_mix_mask
        base_module.PHASE_SYMBOL_XOR_MASK = old_phase_symbol_xor_mask
        base_module.PHASE_REPAIR = old_phase_repair
        base_module.PHASE_CANDIDATE_DECODING = old_phase_candidate_decoding


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
