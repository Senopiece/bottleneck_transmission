import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from ._interface import Config, Estimator, Packet, Message, Protocol, Sampler

# TODO: optimize computation with numba

# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 0

# ---------------------------------------------------------------------------
# Helper utilities shared by sampler and estimator
# ---------------------------------------------------------------------------


def _int_to_bits(value: int, length: int) -> np.ndarray:
    """Return ``length`` bits (MSB first) representing ``value``."""
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        shift = length - 1 - i
        bits[i] = (value >> shift) & 1
    return bits


def _bits_to_int(bits: Iterable[int]) -> int:
    """Pack iterable of bits (MSB first) into an integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


@dataclass(frozen=True)
class _SchemeLayout:
    header_bits: int
    symbol_bits: int
    k: int


def _select_layout(n: int, message_bitsize: int) -> _SchemeLayout:
    """
    Find the smallest header / source symbol configuration that can transfer
    ``message_bitsize`` bits over packets of length ``n``.
    """
    if n <= 0:
        raise ValueError("Packet width n must be positive")

    best_layout: _SchemeLayout | None = None
    best_score: tuple[int, int, int] | None = None

    for header_bits in range(0, n):
        symbol_bits = n - header_bits
        if symbol_bits <= 0:
            continue

        k_required = (
            1 if message_bitsize == 0 else math.ceil(message_bitsize / symbol_bits)
        )
        seed_space = max(1, 1 << header_bits)
        if k_required > seed_space:
            continue

        layout = _SchemeLayout(header_bits, symbol_bits, k_required)
        score = (header_bits, k_required, -symbol_bits)
        if best_score is None or score < best_score:
            best_score = score
            best_layout = layout

    if best_layout is None:
        raise ValueError(
            f"Cannot encode payload of {message_bitsize} bits with packets of width {n}"
        )
    return best_layout


def _robust_soliton_cdf(k: int, c: float = 0.1, delta: float = 0.05) -> list[float]:
    """Robust soliton distribution (cumulative) used to sample degrees."""
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

    cdf = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    cdf[-1] = 1.0  # ensure numerical stability
    return cdf


def _subset_from_seed(
    seed: int, k: int, cdf: list[float], singleton_limit: int | None = None
) -> list[int]:
    """Deterministically derive the neighbor set for a packet from ``seed``."""
    if k <= 0:
        return []

    limit = k if singleton_limit is None else max(0, min(singleton_limit, k))
    if limit and seed < limit:
        return [seed]

    rng = random.Random(seed)
    draw = rng.random()
    degree = 1
    for idx, threshold in enumerate(cdf):
        if draw <= threshold:
            degree = idx + 1
            break
    degree = max(1, min(degree, k))

    if degree == k:
        return list(range(k))

    subset = rng.sample(range(k), degree)
    subset.sort()
    return subset


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _normalize_payload(payload: Message, message_bitsize: int) -> np.ndarray:
    bits = np.array(payload, dtype=np.uint8).reshape(-1) & 1
    if bits.size != message_bitsize:
        raise ValueError(
            f"Expected payload of {message_bitsize} bits, got {bits.size} bits"
        )
    return bits


def _message_bitsize_to_symbols(
    bits: np.ndarray, k: int, symbol_bits: int
) -> np.ndarray:
    """Map payload bits into ``k`` symbols, left-padding with zeros."""
    if symbol_bits == 0:
        return np.zeros((k, 0), dtype=np.uint8)

    total_bits = k * symbol_bits
    if bits.size > total_bits:
        raise ValueError("payload exceeds available symbol capacity")

    if bits.size == total_bits:
        flat = bits
    else:
        flat = np.zeros(total_bits, dtype=np.uint8)
        if bits.size:
            flat[-bits.size :] = bits
    return flat.reshape((k, symbol_bits))


def _symbols_to_payload(symbols: list[np.ndarray], message_bitsize: int) -> np.ndarray:
    """Recover payload bits from flattened symbol list."""
    if message_bitsize == 0:
        return np.zeros(0, dtype=np.bool_)
    if not symbols:
        raise ValueError("no symbols to reconstruct payload")

    if symbols[0].size == 0:
        return np.zeros(message_bitsize, dtype=np.bool_)

    flat = np.concatenate(symbols)
    if flat.size < message_bitsize:
        raise ValueError("not enough symbol bits to recover payload")
    return flat[-message_bitsize:].astype(np.bool_, copy=False)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class _LTSamplerState:
    """Classical LT fountain encoder with metadata headers sized per payload."""

    def __init__(
        self,
        packet_bitsize: int,
        message_bitsize: int,
        payload: Message,
        layout: _SchemeLayout,
    ):
        self.n = packet_bitsize
        self.message_bitsize = message_bitsize
        self.header_bits = layout.header_bits
        self.symbol_bits = layout.symbol_bits
        self.k = layout.k

        self.degree_cdf = _robust_soliton_cdf(self.k)
        message_bitsize_arr = _normalize_payload(payload, message_bitsize)
        payload_seed = _bits_to_int(message_bitsize_arr)
        self.source_symbols = _message_bitsize_to_symbols(
            message_bitsize_arr, self.k, self.symbol_bits
        )
        self.seed_space = max(1, 1 << self.header_bits)

        seed_material = (
            (payload_seed << 16)
            ^ (packet_bitsize << 8)
            ^ (self.k << 4)
            ^ (self.header_bits << 2)
            ^ 0xA5F152C3
        )
        self.rng = random.Random(seed_material)

    def _xor_subset(self, subset: list[int]) -> np.ndarray:
        if self.symbol_bits == 0:
            return np.zeros(0, dtype=np.uint8)
        acc = np.zeros(self.symbol_bits, dtype=np.uint8)
        for idx in subset:
            acc ^= self.source_symbols[idx]
        return acc

    def generate(self) -> Packet:
        seed = self.rng.randrange(self.seed_space)
        header = _int_to_bits(seed, self.header_bits)
        subset = _subset_from_seed(
            seed, self.k, self.degree_cdf, singleton_limit=self.k
        )
        payload = self._xor_subset(subset)
        if self.header_bits == 0:
            packet = payload
        elif self.symbol_bits == 0:
            packet = header
        else:
            packet = np.concatenate([header, payload])
        return packet.astype(np.bool_, copy=False)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class _LTEstimatorState:
    """Peeling decoder for the LT stream."""

    def __init__(
        self, packet_bitsize: int, message_bitsize: int, layout: _SchemeLayout
    ):
        self.n = packet_bitsize
        self.message_bitsize = message_bitsize
        self.header_bits = layout.header_bits
        self.symbol_bits = layout.symbol_bits
        self.k = layout.k
        self.degree_cdf = _robust_soliton_cdf(self.k)

        self.symbols: list[np.ndarray | None] = [None for _ in range(self.k)]
        self.pending: list[tuple[set[int], np.ndarray]] = []
        self.recovered_payload: np.ndarray | None = (
            np.zeros(0, dtype=np.bool_) if self.message_bitsize == 0 else None
        )

    def _propagate_pending(self):
        if not self.pending:
            return

        progress = True
        while progress:
            progress = False
            new_pending: list[tuple[set[int], np.ndarray]] = []
            for subset, payload in self.pending:
                if not subset:
                    continue
                subset = set(subset)
                payload = payload.copy()

                for idx in list(subset):
                    sym = self.symbols[idx]
                    if sym is not None:
                        payload ^= sym
                        subset.remove(idx)

                if not subset:
                    continue

                if len(subset) == 1:
                    idx = next(iter(subset))
                    if self.symbols[idx] is None:
                        self.symbols[idx] = payload.copy()
                        progress = True
                    continue

                new_pending.append((subset, payload))

            self.pending = new_pending

    def progress(self) -> float:
        if self.recovered_payload is not None:
            return 1.0
        if self.k <= 0:
            return 0.0
        known = sum(1 for sym in self.symbols if sym is not None)
        return min(1.0, known / self.k)

    def _try_finalize(self) -> np.ndarray | None:
        if self.recovered_payload is not None:
            return self.recovered_payload
        symbols: list[np.ndarray] = []
        for sym in self.symbols:
            if sym is None:
                return None
            symbols.append(sym)
        self.recovered_payload = _symbols_to_payload(symbols, self.message_bitsize)
        return self.recovered_payload

    def feed(self, data: Packet | None) -> Message | None:
        if self.recovered_payload is not None:
            return self.recovered_payload
        if self.message_bitsize == 0:
            self.recovered_payload = np.zeros(0, dtype=np.bool_)
            return self.recovered_payload
        if data is None:
            return None

        packet = np.array(data, dtype=np.uint8).reshape(-1) & 1
        if packet.shape != (self.n,):
            raise ValueError("packet shape mismatch")

        seed_bits = packet[: self.header_bits]
        packet_payload = packet[self.header_bits :]
        seed = _bits_to_int(seed_bits.tolist())
        subset = _subset_from_seed(
            seed, self.k, self.degree_cdf, singleton_limit=self.k
        )

        residual = packet_payload.copy()
        unknown = []
        for idx in subset:
            sym = self.symbols[idx]
            if sym is None:
                unknown.append(idx)
            else:
                residual ^= sym

        if not unknown:
            return self._try_finalize()

        if len(unknown) == 1:
            target = unknown[0]
            self.symbols[target] = residual.copy()
            self._propagate_pending()
            return self._try_finalize()

        self.pending.append((set(unknown), residual.copy()))
        self._propagate_pending()
        return self._try_finalize()


# ---------------------------------------------------------------------------
# Protocol interface
# ---------------------------------------------------------------------------


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = int(config.packet_bitsize)
    message_bitsize = int(config.message_bitsize)
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    _max_message_bitsize = max_message_bitsize(packet_bitsize)
    if message_bitsize > _max_message_bitsize:
        raise ValueError(
            f"Cannot encode {message_bitsize} payload bits with packet size {packet_bitsize} "
            f"(max {_max_message_bitsize})"
        )

    layout = _select_layout(packet_bitsize, message_bitsize)

    def make_sampler(payload: Message) -> Sampler:
        sampler_state = _LTSamplerState(
            packet_bitsize, message_bitsize, payload, layout
        )
        while True:
            yield sampler_state.generate()

    def make_estimator() -> Estimator:
        estimator_state = _LTEstimatorState(packet_bitsize, message_bitsize, layout)
        packet = yield estimator_state.progress()
        while True:
            recovered = estimator_state.feed(packet)
            if recovered is not None:
                return recovered
            packet = yield estimator_state.progress()

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    """Return the maximum payload size (in bits) for packets of the given size."""
    return 2 ** (packet_bitsize - 1)


def _lt_expected_received_symbols(
    k: int,
    *,
    # For small k, use explicit expected "received symbols needed"
    # Populate this to match your implementation.
    small_k_table: Dict[int, int] | None = None,
    # Fallback asymptotic proxy params
    c: float = 0.1,
    delta: float = 0.05,
    # Extra constant overhead even for larger k (implementation/decoder constants)
    extra: int = 0,
) -> int:
    if k <= 0:
        return 0

    if small_k_table is not None and k in small_k_table:
        return max(k, int(small_k_table[k]))

    # Robust-Soliton-like proxy (asymptotic)
    if k < 3:
        return k + extra

    s = c * math.sqrt(k) * math.log(k / delta)
    if s <= 1e-12:
        return k + extra

    overhead = int(math.ceil(s * math.log(s / delta)))
    return max(k, k + overhead + extra)


def _expected_steps_to_get_r_successes_ge(
    r: int,
    pGB: float,
    pBG: float,
    pG: float,
    pB: float,
    start_in_good: bool,
) -> float:
    if r <= 0:
        return 0.0
    qG = pGB  # good -> bad
    qB = pBG  # bad  -> good

    FG_prev = 0.0
    FB_prev = 0.0

    for _ in range(1, r + 1):
        a11 = 1.0 - (1.0 - pG) * (1.0 - qG)
        a12 = -(1.0 - pG) * qG
        b1 = 1.0 + pG * ((1.0 - qG) * FG_prev + qG * FB_prev)

        a21 = -(1.0 - pB) * qB
        a22 = 1.0 - (1.0 - pB) * (1.0 - qB)
        b2 = 1.0 + pB * ((1.0 - qB) * FB_prev + qB * FG_prev)

        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-15:
            return float("inf")

        FG = (b1 * a22 - b2 * a12) / det
        FB = (a11 * b2 - a21 * b1) / det
        FG_prev, FB_prev = FG, FB

    return FG_prev if start_in_good else FB_prev


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],  # pGB, pBG, pG, pB
    packet_bitsize: int,
    message_bitsize: int,
) -> float:
    pGB, pBG, pG, pB = gilbert_eliott_k

    if message_bitsize <= 0:
        return 0.0
    if packet_bitsize <= 0:
        raise ValueError("packet_bitsize must be > 0")

    n = float(packet_bitsize)
    M = float(message_bitsize)

    # ---- continuous header / symbol split ----
    logM = math.log2(M)
    denom = n - logM
    if denom <= 0.0:
        return float("inf")

    symbol_bits = n - logM + math.log2(denom)
    if symbol_bits <= 0.0:
        return float("inf")

    # continuous k
    k = M / symbol_bits

    # ---- LT decoding overhead (continuous, implementation-aware) ----
    alpha = max(0.0, math.log(k) - 1.0)  # redundancy due to singleton saturation
    beta = 1.0  # peeling stall penalty

    useful_packets = k * (1.0 + alpha) + beta * math.log(k + 1.0)

    # ---- GE stationary success probability ----
    denom = pBG + pGB
    if denom <= 0.0:
        return float("inf")

    piG = pBG / denom
    piB = pGB / denom

    p_eff = piG * pG + piB * pB
    if p_eff <= 0.0:
        return float("inf")

    # ---- expected packets until reconstruction ----
    return useful_packets / p_eff
