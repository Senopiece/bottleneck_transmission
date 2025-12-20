import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ._interface import Producer, Recoverer


# ---------------------------------------------------------------------------
# Helper utilities shared by producer and recoverer
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


def _payload_bits_for_d(d: int) -> int:
    """Return how many bits are needed to encode ``d`` payload states."""
    if d <= 1:
        return 0
    return math.ceil(math.log2(d))


def _max_payload_bits_for_n(n: int) -> int:
    """Best-case payload capacity (bits) for packets of width ``n``."""
    if n <= 0:
        return 0
    best = 0
    for header_bits in range(0, n):
        symbol_bits = n - header_bits
        if symbol_bits <= 0:
            continue
        k_max = max(1, 1 << header_bits)
        capacity = k_max * symbol_bits
        if capacity > best:
            best = capacity
    return best


def _select_layout(n: int, payload_bits: int) -> _SchemeLayout:
    """
    Find the smallest header / source symbol configuration that can transfer
    ``payload_bits`` bits over packets of length ``n``.
    """
    if n <= 0:
        raise ValueError("Packet width n must be positive")

    best_layout: _SchemeLayout | None = None
    best_score: tuple[int, int, int] | None = None

    for header_bits in range(0, n):
        symbol_bits = n - header_bits
        if symbol_bits <= 0:
            continue

        k_required = 1 if payload_bits == 0 else math.ceil(payload_bits / symbol_bits)
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
            f"Cannot encode payload of {payload_bits} bits with packets of width {n}"
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


def _index_to_symbols(index: int, k: int, bits_per_symbol: int) -> np.ndarray:
    """Convert payload index into ``k`` binary source blocks."""
    if bits_per_symbol == 0:
        return np.zeros((k, 0), dtype=np.uint8)
    total_bits = k * bits_per_symbol
    if index < 0 or index >= 1 << total_bits:
        raise ValueError(f"index={index} is out of range for {k} symbols")
    flat_bits = _int_to_bits(index, total_bits)
    return flat_bits.reshape((k, bits_per_symbol))


def _symbols_to_index(symbols: list[np.ndarray]) -> int:
    """Flatten symbol list back into payload integer."""
    if not symbols:
        return 0
    concatenated = np.concatenate(symbols) if symbols[0].size else np.zeros(0)
    return _bits_to_int(concatenated.tolist())


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


class LTProducer(Producer):
    """Classical LT fountain encoder with metadata headers sized per payload."""

    def __init__(self, n: int, index: int, d: int):
        if index < 0 or index >= d:
            raise ValueError(f"index must be in [0, {d})")

        self.n = n
        self.d = d
        self.payload_bits = _payload_bits_for_d(d)
        max_payload_bits = _max_payload_bits_for_n(n)
        if self.payload_bits > max_payload_bits:
            raise ValueError(
                f"Cannot encode {self.payload_bits} payload bits with n={n} (max {max_payload_bits})"
            )

        layout = _select_layout(n, self.payload_bits)
        self.header_bits = layout.header_bits
        self.symbol_bits = layout.symbol_bits
        self.k = layout.k

        self.degree_cdf = _robust_soliton_cdf(self.k)
        self.source_symbols = _index_to_symbols(index, self.k, self.symbol_bits)
        self.seed_space = max(1, 1 << self.header_bits)

        seed_material = (
            (index << 16)
            ^ (n << 8)
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

    def generate(self) -> np.ndarray:
        seed = self.rng.randrange(self.seed_space)
        header = _int_to_bits(seed, self.header_bits)
        subset = _subset_from_seed(
            seed, self.k, self.degree_cdf, singleton_limit=self.k
        )
        payload = self._xor_subset(subset)
        if self.header_bits == 0:
            return payload.astype(np.uint8)
        if self.symbol_bits == 0:
            return header
        return np.concatenate([header, payload]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Recoverer
# ---------------------------------------------------------------------------


class LTRecoverer(Recoverer):
    """Peeling decoder for the LT stream."""

    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self.payload_bits = _payload_bits_for_d(d)

        layout = _select_layout(n, self.payload_bits)
        self.header_bits = layout.header_bits
        self.symbol_bits = layout.symbol_bits
        self.k = layout.k
        self.degree_cdf = _robust_soliton_cdf(self.k)

        self.symbols: list[np.ndarray | None] = [None for _ in range(self.k)]
        self.pending: list[tuple[set[int], np.ndarray]] = []
        self.recovered_index: int | None = 0 if self.payload_bits == 0 else None

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

    def _try_finalize(self) -> int | None:
        if self.recovered_index is not None:
            return self.recovered_index
        if any(sym is None for sym in self.symbols):
            return None
        symbols = [self.symbols[i] for i in range(self.k)]
        self.recovered_index = _symbols_to_index(symbols)
        return self.recovered_index

    def feed(self, data: np.ndarray | None) -> int | None:
        if self.recovered_index is not None:
            return self.recovered_index
        if self.payload_bits == 0:
            return self.recovered_index
        if data is None:
            return None

        packet = np.array(data, dtype=np.uint8) & 1
        assert packet.shape == (self.n,), "packet shape mismatch"

        seed_bits = packet[: self.header_bits]
        payload_bits = packet[self.header_bits :]
        seed = _bits_to_int(seed_bits.tolist())
        subset = _subset_from_seed(
            seed, self.k, self.degree_cdf, singleton_limit=self.k
        )

        residual = payload_bits.copy()
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
# Exported hooks for the benchmark harness
# ---------------------------------------------------------------------------


def override_D(n: int) -> int:
    max_bits = _max_payload_bits_for_n(n)
    return 1 << max_bits if max_bits > 0 else 1


def producer_constructor(index: int, n: int, d: int) -> Producer:
    if d > override_D(n):
        raise ValueError(f"d={d} exceeds maximum supported payload for n={n}")
    return LTProducer(n, index, d)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    if d > override_D(n):
        raise ValueError(f"d={d} exceeds maximum supported payload for n={n}")
    return LTRecoverer(n, d)
