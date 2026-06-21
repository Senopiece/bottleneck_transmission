from typing import Tuple

from . import _repair_core_fly as base_module
from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils import sparce

# Decoder-heavy reliability profile with no guard packets.
# It keeps packet payload capacity unchanged and waits for a small number of
# extra received packets after first full peeling, then repairs symbols against
# the equation bank.

USE_TUNED_SOLITON = False
SOLITON_C = 0.02
SOLITON_DELTA = 0.65
SEGMENT_LENGTH_BONUS = 0.0
SEGMENT_PROBES = 4
RECENT_WINDOW_MARGIN = 6

PHASE_MIXING = False
PHASE_MIX_MODE = "symbol_xor"
PHASE_SYMBOL_XOR_MASK = 0b0011
PHASE_REPAIR = False
PHASE_CANDIDATE_DECODING = False

SMALL_REPAIR_EXTRA_PACKETS = 8
LARGE_REPAIR_EXTRA_PACKETS = 0
REPAIR_PASSES = 8
SINGLETON_CONFIRMATIONS = 1
SMALL_RANSAC_REPAIR_ITERATIONS = 40
LARGE_RANSAC_REPAIR_ITERATIONS = 64
SMALL_RANSAC_MAJORITY_TOP = 9
SMALL_RANSAC_MAJORITY_MARGIN = 2
LARGE_RANSAC_MAJORITY_TOP = 5
LARGE_RANSAC_MAJORITY_MARGIN = 1


def _repair_extra_packets(config: Config) -> int:
    return (
        SMALL_REPAIR_EXTRA_PACKETS
        if int(config.message_bitsize) <= 128
        else LARGE_REPAIR_EXTRA_PACKETS
    )


def _ransac_majority_top(config: Config) -> int:
    return (
        SMALL_RANSAC_MAJORITY_TOP
        if int(config.message_bitsize) <= 128
        else LARGE_RANSAC_MAJORITY_TOP
    )


def _ransac_repair_iterations(config: Config) -> int:
    return (
        SMALL_RANSAC_REPAIR_ITERATIONS
        if int(config.message_bitsize) <= 128
        else LARGE_RANSAC_REPAIR_ITERATIONS
    )


def _ransac_majority_margin(config: Config) -> int:
    return (
        SMALL_RANSAC_MAJORITY_MARGIN
        if int(config.message_bitsize) <= 128
        else LARGE_RANSAC_MAJORITY_MARGIN
    )


def _apply_settings(config: Config) -> dict[str, object]:
    old_settings = {
        "robust_soliton_cdf": base_module.robust_soliton_cdf,
        "SEGMENT_LENGTH_BONUS": base_module.SEGMENT_LENGTH_BONUS,
        "SEGMENT_PROBES": base_module.SEGMENT_PROBES,
        "RECENT_WINDOW_MARGIN": base_module.RECENT_WINDOW_MARGIN,
        "PHASE_MIXING": base_module.PHASE_MIXING,
        "PHASE_MIX_MODE": base_module.PHASE_MIX_MODE,
        "PHASE_SYMBOL_XOR_MASK": base_module.PHASE_SYMBOL_XOR_MASK,
        "PHASE_REPAIR": base_module.PHASE_REPAIR,
        "PHASE_CANDIDATE_DECODING": base_module.PHASE_CANDIDATE_DECODING,
        "EQUATION_BANK_REPAIR": base_module.EQUATION_BANK_REPAIR,
        "REPAIR_EXTRA_PACKETS": base_module.REPAIR_EXTRA_PACKETS,
        "REPAIR_PASSES": base_module.REPAIR_PASSES,
        "SINGLETON_CONFIRMATIONS": base_module.SINGLETON_CONFIRMATIONS,
        "RANSAC_REPAIR_ITERATIONS": base_module.RANSAC_REPAIR_ITERATIONS,
        "RANSAC_MAJORITY_TOP": base_module.RANSAC_MAJORITY_TOP,
        "RANSAC_MAJORITY_MARGIN": base_module.RANSAC_MAJORITY_MARGIN,
    }

    def tuned_robust_soliton_cdf(k: int) -> list[float]:
        return sparce.robust_soliton_cdf(k, c=SOLITON_C, delta=SOLITON_DELTA)

    if USE_TUNED_SOLITON:
        base_module.robust_soliton_cdf = tuned_robust_soliton_cdf
    base_module.SEGMENT_LENGTH_BONUS = SEGMENT_LENGTH_BONUS
    base_module.SEGMENT_PROBES = SEGMENT_PROBES
    base_module.RECENT_WINDOW_MARGIN = RECENT_WINDOW_MARGIN
    base_module.PHASE_MIXING = PHASE_MIXING
    base_module.PHASE_MIX_MODE = PHASE_MIX_MODE
    base_module.PHASE_SYMBOL_XOR_MASK = PHASE_SYMBOL_XOR_MASK
    base_module.PHASE_REPAIR = PHASE_REPAIR
    base_module.PHASE_CANDIDATE_DECODING = PHASE_CANDIDATE_DECODING
    base_module.EQUATION_BANK_REPAIR = True
    base_module.REPAIR_EXTRA_PACKETS = _repair_extra_packets(config)
    base_module.REPAIR_PASSES = REPAIR_PASSES
    base_module.SINGLETON_CONFIRMATIONS = SINGLETON_CONFIRMATIONS
    base_module.RANSAC_REPAIR_ITERATIONS = _ransac_repair_iterations(config)
    base_module.RANSAC_MAJORITY_TOP = _ransac_majority_top(config)
    base_module.RANSAC_MAJORITY_MARGIN = _ransac_majority_margin(config)

    return old_settings


def _restore_settings(old_settings: dict[str, object]) -> None:
    for name, value in old_settings.items():
        setattr(base_module, name, value)


def create_protocol(config: Config) -> Protocol:
    old_settings = _apply_settings(config)
    try:
        base_protocol = base_module.create_protocol(config)
    finally:
        _restore_settings(old_settings)

    def make_sampler(message: Message) -> Sampler:
        base_sampler = base_protocol.make_sampler(message)
        while True:
            old = _apply_settings(config)
            try:
                yield next(base_sampler)
            finally:
                _restore_settings(old)

    def make_estimator() -> Estimator:
        base_estimator = base_protocol.make_estimator()
        old = _apply_settings(config)
        try:
            progress = next(base_estimator)
        finally:
            _restore_settings(old)

        while True:
            packet = yield progress
            old = _apply_settings(config)
            try:
                progress = base_estimator.send(packet)
            except StopIteration as exc:
                return exc.value
            finally:
                _restore_settings(old)

    return Protocol(make_sampler=make_sampler, make_estimator=make_estimator)


def max_message_bitsize(packet_bitsize: int) -> int:
    return base_module.max_message_bitsize(packet_bitsize)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
