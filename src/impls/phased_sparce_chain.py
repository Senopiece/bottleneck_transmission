import math
from typing import Dict, List, Set, Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.sparce import (
    MAX_PREFIX_INPUT_BITS,
    pack_prefix_symbols,
    peel_add_equation,
    peel_propagate,
    robust_soliton_cdf,
    subset_from_x,
)

# Domain:
# skip_probability: [0, 1)
# corruption_probability: 0
# skip_observation: 1.0

# Number of previous (n-1)-bit symbols packed into x for f(x).
PREFIX = 1


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = int(config.packet_bitsize)
    message_bitsize = int(config.message_bitsize)

    if packet_bitsize <= 1:
        raise ValueError("packet_bitsize must be >= 2 to reserve a phase bit")
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    N = packet_bitsize
    z = N - 1  # symbol bits per packet (phase bit excluded)
    symbol_q = 1 << z
    symbol_mask = symbol_q - 1

    input_bits = PREFIX * z
    if input_bits > MAX_PREFIX_INPUT_BITS:
        raise ValueError(
            "phased_sparce_chain is tailored for small packet sizes: "
            f"PREFIX*(packet_bitsize-1) must be <= {MAX_PREFIX_INPUT_BITS}, got {input_bits}"
        )
    state_q = 1 << input_bits

    max_bits = max_message_bitsize(packet_bitsize)
    if message_bitsize > max_bits:
        raise ValueError(
            f"message_bitsize too large for phased_sparce_chain: max {max_bits} for packet_bitsize={packet_bitsize}"
        )

    half_bits_a = (message_bitsize + 1) // 2
    half_bits_b = message_bitsize - half_bits_a

    k1 = math.ceil(half_bits_a / z) if half_bits_a > 0 else 0
    k2 = math.ceil(half_bits_b / z) if half_bits_b > 0 else 0

    cdf_a = robust_soliton_cdf(k1) if k1 > 0 else [1.0]
    cdf_b = robust_soliton_cdf(k2) if k2 > 0 else [1.0]

    # TODO: for small packet sizes it may be beneficial to check several salts and pick the best among N candidates
    salt_a = 0x9E3779B97F4A7C15
    salt_b = 0xD1B54A32D192ED03

    def make_sampler(message: Message) -> Sampler:
        msg_a = message[:half_bits_a]
        msg_b = message[half_bits_a:]

        symbols_a = (
            make_message_vector(msg_a, k1, symbol_q)
            if k1 > 0
            else np.empty(0, dtype=np.uint16)
        )
        symbols_b = (
            make_message_vector(msg_b, k2, symbol_q)
            if k2 > 0
            else np.empty(0, dtype=np.uint16)
        )

        def f1(x: int) -> int:
            if k1 == 0:
                return 0
            subset = subset_from_x(x, k1, cdf_a, salt_a, singleton_limit=k1)
            y = 0
            for idx in subset:
                y ^= int(symbols_a[idx])
            return y & symbol_mask

        def f2(x: int) -> int:
            if k2 == 0:
                return 0
            subset = subset_from_x(x, k2, cdf_b, salt_b, singleton_limit=k2)
            y = 0
            for idx in subset:
                y ^= int(symbols_b[idx])
            return y & symbol_mask

        def x_to_symbols(x: int) -> List[int]:
            symbols = [0 for _ in range(PREFIX)]
            curr = int(x)
            for i in range(PREFIX - 1, -1, -1):
                symbols[i] = curr & symbol_mask
                curr >>= z
            return symbols

        def phased_output(symbol: int, is_phase_b: bool):
            value = symbol + (symbol_q if is_phase_b else 0)
            return uint16_to_bool_array(np.uint16(value), N)

        singleton_a = list(range(min(k1, state_q)))
        singleton_b = list(range(min(k2, state_q)))
        last_output: List[Tuple[int, bool] | None] = [None]

        def make_output(symbol: int, is_phase_b: bool):
            symbol = int(symbol) & symbol_mask
            last_output[0] = (symbol, is_phase_b)
            return phased_output(symbol, is_phase_b)

        def bridge_output(is_phase_b: bool):
            symbol = 0
            last = last_output[0]
            if last is not None and last == (symbol, is_phase_b):
                symbol = 1 if symbol_q > 1 else 0
            return make_output(symbol, is_phase_b)

        def emit_equation_segment(x: int, use_a: bool):
            src_phase_b = not use_a  # use_a: A->B, else B->A
            dst_phase_b = not src_phase_b
            prefix_symbols = x_to_symbols(x)
            y = f1(x) if use_a else f2(x)

            prefix_phases = [
                bool(src_phase_b ^ ((PREFIX - 1 - i) & 1)) for i in range(PREFIX)
            ]
            seq_symbols = prefix_symbols + [y]
            seq_phases = prefix_phases + [dst_phase_b]

            # Always force a local reset marker regardless of previous stream context.
            first_symbol = seq_symbols[0]
            first_phase_b = seq_phases[0]
            yield make_output(first_symbol, first_phase_b)
            yield make_output(first_symbol, first_phase_b)

            for symbol, phase_b in zip(seq_symbols[1:], seq_phases[1:]):
                yield make_output(symbol, phase_b)

        while True:
            if k1 > 0:
                for x in singleton_a:
                    for packet in emit_equation_segment(x, use_a=True):
                        yield packet
            if k1 > 0 and k2 > 0:
                # Bridge A->B with a same-phase reset (A segments end in phase B=True).
                yield bridge_output(True)
            if k2 > 0:
                for x in singleton_b:
                    for packet in emit_equation_segment(x, use_a=False):
                        yield packet
            if k1 > 0 and k2 > 0:
                # Bridge B->A with a same-phase reset (B segments end in phase B=False).
                yield bridge_output(False)
            if k1 == 0 and k2 == 0:
                yield make_output(0, False)

    def make_estimator() -> Estimator:
        symbols_a: List[int | None] = [None for _ in range(k1)]
        symbols_b: List[int | None] = [None for _ in range(k2)]
        pending_a: List[Tuple[Set[int], int]] = []
        pending_b: List[Tuple[Set[int], int]] = []

        seen_a: Dict[int, int] = {}
        seen_b: Dict[int, int] = {}

        run_symbols: List[int] = []
        run_phases: List[bool] = []
        max_run_keep = PREFIX + 1
        pending_edge: Tuple[int, int, bool, bool] | None = None

        total_k = k1 + k2

        def progress() -> float:
            if total_k == 0:
                return 1.0
            known = sum(1 for v in symbols_a if v is not None) + sum(
                1 for v in symbols_b if v is not None
            )
            return known / total_k

        def maybe_done() -> np.ndarray | None:
            if any(v is None for v in symbols_a) or any(v is None for v in symbols_b):
                return None

            parts = []
            if k1 > 0:
                vec_a = np.array(symbols_a, dtype=np.uint16)
                parts.append(message_from_message_vector(vec_a, half_bits_a, symbol_q))
            else:
                parts.append(np.zeros(0, dtype=np.bool_))

            if k2 > 0:
                vec_b = np.array(symbols_b, dtype=np.uint16)
                parts.append(message_from_message_vector(vec_b, half_bits_b, symbol_q))
            else:
                parts.append(np.zeros(0, dtype=np.bool_))

            return np.concatenate(parts)

        def commit_pending_edge() -> None:
            nonlocal pending_edge
            if pending_edge is None:
                return

            x, y, src_phase, dst_phase = pending_edge
            pending_edge = None

            if (not src_phase) and dst_phase:
                if k1 > 0 and x not in seen_a:
                    seen_a[x] = y
                    peel_add_equation(x, y, k1, cdf_a, salt_a, symbols_a, pending_a)
                    peel_propagate(symbols_a, pending_a)
            elif src_phase and (not dst_phase):
                if k2 > 0 and x not in seen_b:
                    seen_b[x] = y
                    peel_add_equation(x, y, k2, cdf_b, salt_b, symbols_b, pending_b)
                    peel_propagate(symbols_b, pending_b)

        while True:
            done = maybe_done()
            if done is not None:
                return done

            packet = yield progress()

            if packet is None:
                pending_edge = None
                run_symbols.clear()
                run_phases.clear()
                continue

            curr_phase = bool(packet[0])
            curr_symbol = int(bool_array_to_uint16(packet) & symbol_mask)

            if run_phases and curr_phase == run_phases[-1]:
                is_duplicate_reset = curr_symbol == run_symbols[-1]
                if pending_edge is not None:
                    if is_duplicate_reset:
                        pending_edge = None
                    else:
                        commit_pending_edge()
                run_symbols[:] = [curr_symbol]
                run_phases[:] = [curr_phase]
                continue

            run_symbols.append(curr_symbol)
            run_phases.append(curr_phase)

            if len(run_symbols) > max_run_keep:
                run_symbols[:] = run_symbols[-max_run_keep:]
                run_phases[:] = run_phases[-max_run_keep:]

            if len(run_symbols) < (PREFIX + 1):
                continue

            commit_pending_edge()

            x = pack_prefix_symbols(run_symbols[-(PREFIX + 1) : -1], z)
            y = run_symbols[-1]
            src_phase = run_phases[-2]
            dst_phase = run_phases[-1]

            pending_edge = (x, y, src_phase, dst_phase)

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if packet_bitsize <= 1:
        return 0
    z = packet_bitsize - 1
    input_bits = PREFIX * z
    if input_bits > MAX_PREFIX_INPUT_BITS:
        return 0
    return 2 * z * (1 << input_bits)


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],  # pGB, pBG, pG, pB
    packet_bitsize: int,
    message_bitsize: int,
):
    return None
