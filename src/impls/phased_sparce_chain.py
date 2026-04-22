import math
import random
from typing import Dict, List, Set, Tuple

import numpy as np

from ._interface import Config, Estimator, Message, Protocol, Sampler
from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)

# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1.0

# Number of previous (n-1)-bit symbols packed into x for f(x).
PREFIX = 1

# Keep state-space bounded; this implementation is tailored for small packet sizes.
MAX_PREFIX_INPUT_BITS = 16


def _robust_soliton_cdf(k: int, c: float = 0.1, delta: float = 0.05) -> List[float]:
    if k <= 0:
        return [1.0]

    ideal = [0.0 for _ in range(k)]
    ideal[0] = 1.0 / k
    for d in range(2, k + 1):
        ideal[d - 1] = 1.0 / (d * (d - 1))

    R = c * math.log(k / delta) * math.sqrt(k)
    if R < 1.0:
        R = 1.0
    k_over_R = max(1, int(math.floor(k / R)))

    tau = [0.0 for _ in range(k)]
    for d in range(1, k + 1):
        if d < k_over_R:
            tau[d - 1] = R / (d * k)
        elif d == k_over_R:
            tau[d - 1] = R * math.log(R / delta) / k

    normalizer = sum(ideal[i] + tau[i] for i in range(k))
    probs = [(ideal[i] + tau[i]) / normalizer for i in range(k)]

    # Small-k tuning: force a stronger singleton ripple for tiny packets.
    if k <= 16:
        target_p1 = 0.35 if k <= 8 else 0.25
        p1 = probs[0]
        if p1 < target_p1:
            rest_sum = max(1e-12, 1.0 - p1)
            scale = (1.0 - target_p1) / rest_sum
            probs[0] = target_p1
            for i in range(1, k):
                probs[i] *= scale

    cdf: List[float] = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf


def _subset_from_x(
    x: int,
    k: int,
    cdf: List[float],
    salt: int,
    singleton_limit: int,
) -> List[int]:
    if k <= 0:
        return []

    xi = int(x)
    limit = max(0, min(singleton_limit, k))
    if limit > 0 and xi < limit:
        return [xi]

    rng = random.Random((xi ^ salt) & 0xFFFFFFFFFFFFFFFF)
    draw = rng.random()

    degree = 1
    for i, threshold in enumerate(cdf):
        if draw <= threshold:
            degree = i + 1
            break

    degree = max(1, min(degree, k))
    if degree == k:
        return list(range(k))

    subset = rng.sample(range(k), degree)
    subset.sort()
    return subset


def _pack_prefix_symbols(symbols: List[int], z: int) -> int:
    x = 0
    for symbol in symbols:
        x = (x << z) | int(symbol)
    return x


def _peel_add_equation(
    x: int,
    y: int,
    k: int,
    cdf: List[float],
    salt: int,
    symbols: List[int | None],
    pending: List[Tuple[Set[int], int]],
) -> bool:
    if k <= 0:
        return False

    subset = _subset_from_x(x, k, cdf, salt, singleton_limit=k)
    rhs = int(y)

    unknown: Set[int] = set()
    for idx in subset:
        known = symbols[idx]
        if known is None:
            unknown.add(idx)
        else:
            rhs ^= known

    if not unknown:
        return False

    changed = False
    if len(unknown) == 1:
        idx = next(iter(unknown))
        if symbols[idx] is None:
            symbols[idx] = rhs
            changed = True
        return changed

    pending.append((unknown, rhs))
    return changed


def _peel_propagate(
    symbols: List[int | None], pending: List[Tuple[Set[int], int]]
) -> bool:
    changed_any = False

    progressed = True
    while progressed:
        progressed = False
        next_pending: List[Tuple[Set[int], int]] = []

        for unknown, rhs in pending:
            curr_unknown = set(unknown)
            curr_rhs = int(rhs)

            for idx in list(curr_unknown):
                known = symbols[idx]
                if known is not None:
                    curr_rhs ^= known
                    curr_unknown.remove(idx)

            if not curr_unknown:
                continue

            if len(curr_unknown) == 1:
                idx = next(iter(curr_unknown))
                if symbols[idx] is None:
                    symbols[idx] = curr_rhs
                    progressed = True
                    changed_any = True
                continue

            next_pending.append((curr_unknown, curr_rhs))

        pending[:] = next_pending

    return changed_any


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

    cdf_a = _robust_soliton_cdf(k1) if k1 > 0 else [1.0]
    cdf_b = _robust_soliton_cdf(k2) if k2 > 0 else [1.0]

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
            subset = _subset_from_x(x, k1, cdf_a, salt_a, singleton_limit=k1)
            y = 0
            for idx in subset:
                y ^= int(symbols_a[idx])
            return y & symbol_mask

        def f2(x: int) -> int:
            if k2 == 0:
                return 0
            subset = _subset_from_x(x, k2, cdf_b, salt_b, singleton_limit=k2)
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
            yield phased_output(first_symbol, first_phase_b)
            yield phased_output(first_symbol, first_phase_b)

            for symbol, phase_b in zip(seq_symbols[1:], seq_phases[1:]):
                yield phased_output(symbol, phase_b)

        while True:
            if k1 > 0:
                for x in singleton_a:
                    for packet in emit_equation_segment(x, use_a=True):
                        yield packet
            if k1 > 0 and k2 > 0:
                # Bridge A->B with a same-phase reset (A segments end in phase B=True).
                yield phased_output(0, True)
            if k2 > 0:
                for x in singleton_b:
                    for packet in emit_equation_segment(x, use_a=False):
                        yield packet
            if k1 > 0 and k2 > 0:
                # Bridge B->A with a same-phase reset (B segments end in phase B=False).
                yield phased_output(0, False)
            if k1 == 0 and k2 == 0:
                yield phased_output(0, False)

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

        while True:
            done = maybe_done()
            if done is not None:
                return done

            packet = yield progress()

            if packet is None:
                run_symbols.clear()
                run_phases.clear()
                continue

            curr_phase = bool(packet[0])
            curr_symbol = int(bool_array_to_uint16(packet) & symbol_mask)

            if run_phases and curr_phase == run_phases[-1]:
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

            x = _pack_prefix_symbols(run_symbols[-(PREFIX + 1) : -1], z)
            y = run_symbols[-1]
            src_phase = run_phases[-2]
            dst_phase = run_phases[-1]

            # A -> B gives equation over first half symbols.
            if (not src_phase) and dst_phase:
                if k1 > 0 and x not in seen_a:
                    seen_a[x] = y
                    _peel_add_equation(x, y, k1, cdf_a, salt_a, symbols_a, pending_a)
                    _peel_propagate(symbols_a, pending_a)
            # B -> A gives equation over second half symbols.
            elif src_phase and (not dst_phase):
                if k2 > 0 and x not in seen_b:
                    seen_b[x] = y
                    _peel_add_equation(x, y, k2, cdf_b, salt_b, symbols_b, pending_b)
                    _peel_propagate(symbols_b, pending_b)

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
