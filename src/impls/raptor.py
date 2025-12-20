import math
import random
from dataclasses import dataclass

import numpy as np

from ._interface import Producer, Recoverer
from .lt import (
    _bits_to_int,
    _index_to_symbols,
    _int_to_bits,
    _payload_bits_for_d,
    _robust_soliton_cdf,
    _subset_from_seed,
    _symbols_to_index,
)


@dataclass(frozen=True)
class _RaptorLayout:
    header_bits: int
    symbol_bits: int
    k: int
    s: int
    h: int

    @property
    def intermediate_symbols(self) -> int:
        return self.k + self.s + self.h


def _precode_parameters(k: int) -> tuple[int, int]:
    """Number of LDPC and HDPC nodes to add on top of ``k`` source symbols."""
    if k <= 0:
        return 0, 0
    s = max(1, int(math.ceil(0.08 * k)))
    h = max(1, int(math.ceil(0.02 * k)))
    return s, h


def _select_layout(n: int, payload_bits: int) -> _RaptorLayout:
    if n <= 0:
        raise ValueError("Packet width n must be positive")

    best_layout: _RaptorLayout | None = None
    best_score: tuple[int, int, int] | None = None

    for header_bits in range(0, n):
        symbol_bits = n - header_bits
        if symbol_bits <= 0:
            continue

        k_required = 1 if payload_bits == 0 else math.ceil(payload_bits / symbol_bits)
        s, h = _precode_parameters(k_required)
        l = k_required + s + h
        seed_space = max(1, 1 << header_bits)
        if l == 0:
            continue
        if k_required > seed_space:
            continue

        layout = _RaptorLayout(header_bits, symbol_bits, k_required, s, h)
        score = (header_bits, k_required, -symbol_bits)
        if best_score is None or score < best_score:
            best_score = score
            best_layout = layout

    if best_layout is None:
        raise ValueError(
            f"Cannot encode payload of {payload_bits} bits with packets of width {n}"
        )
    return best_layout


def _max_payload_bits_for_n(n: int) -> int:
    """Upper bound on payload bits supported by raptor scheme for packet width n."""
    if n <= 0:
        return 0

    best = 0
    for header_bits in range(0, n):
        symbol_bits = n - header_bits
        if symbol_bits <= 0:
            continue
        seed_space = max(1, 1 << header_bits)
        for k_candidate in range(1, seed_space + 1):
            s, h = _precode_parameters(k_candidate)
            l = k_candidate + s + h
            if l == 0 or k_candidate > seed_space:
                continue
            capacity = k_candidate * symbol_bits
            if capacity > best:
                best = capacity
    return best


def _precode_seed(n: int, symbol_bits: int, layout: _RaptorLayout) -> int:
    return (
        (n & 0xFFFF) << 32
        ^ (symbol_bits & 0xFFFF) << 16
        ^ (layout.k & 0xFFFF) << 8
        ^ (layout.s & 0xFF) << 4
        ^ (layout.h & 0xFF)
        ^ 0x57ED5A1B
    )


def _build_precode_graph(n: int, symbol_bits: int, layout: _RaptorLayout) -> dict[int, list[int]]:
    """Return neighbors for each parity node."""
    graph: dict[int, list[int]] = {}
    total = layout.intermediate_symbols
    seed = _precode_seed(n, symbol_bits, layout)
    rng = random.Random(seed)

    for idx in range(layout.k, total):
        if idx < layout.k + layout.s:
            population = layout.k
            if population == 0:
                graph[idx] = []
                continue
            deg = 1 if population == 1 else min(3, population)
            if population < deg:
                deg = population
            neighbors = sorted(rng.sample(range(population), deg))
        else:
            population = idx
            if population == 0:
                graph[idx] = []
                continue
            base = max(3, int(math.log2(layout.k + 1)) + 2)
            deg = min(population, base)
            neighbors = sorted(rng.sample(range(population), deg))
        graph[idx] = neighbors
    return graph


class RaptorProducer(Producer):
    def __init__(self, n: int, index: int, d: int):
        if index < 0 or index >= d:
            raise ValueError(f"index must be in [0, {d})")

        self.n = n
        self.d = d
        self.payload_bits = _payload_bits_for_d(d)

        layout = _select_layout(n, self.payload_bits)
        self.header_bits = layout.header_bits
        self.symbol_bits = layout.symbol_bits
        self.k = layout.k
        self.s = layout.s
        self.h = layout.h
        self.l = layout.intermediate_symbols

        self.degree_cdf = _robust_soliton_cdf(self.l)
        self.source_symbols = _index_to_symbols(index, self.k, self.symbol_bits)
        self.precode_graph = _build_precode_graph(n, self.symbol_bits, layout)
        self.intermediate_symbols = self._build_intermediate_symbols()
        self.seed_space = max(1, 1 << self.header_bits)

        seed_material = (
            (index << 24)
            ^ (n << 16)
            ^ (self.k << 12)
            ^ (self.s << 8)
            ^ (self.h << 4)
            ^ (self.header_bits << 2)
            ^ 0xB16B00B5
        )
        self.rng = random.Random(seed_material)

    def _build_intermediate_symbols(self) -> np.ndarray:
        if self.l <= 0:
            return np.zeros((0, self.symbol_bits), dtype=np.uint8)
        symbols = np.zeros((self.l, self.symbol_bits), dtype=np.uint8)
        if self.k > 0 and self.symbol_bits > 0:
            symbols[: self.k] = self.source_symbols.copy()
        for idx in range(self.k, self.l):
            neighbors = self.precode_graph.get(idx, [])
            acc = np.zeros(self.symbol_bits, dtype=np.uint8)
            for nb in neighbors:
                acc ^= symbols[nb]
            symbols[idx] = acc
        return symbols

    def _xor_subset(self, subset: list[int]) -> np.ndarray:
        if self.symbol_bits == 0:
            return np.zeros(0, dtype=np.uint8)
        acc = np.zeros(self.symbol_bits, dtype=np.uint8)
        for idx in subset:
            acc ^= self.intermediate_symbols[idx]
        return acc

    def generate(self) -> np.ndarray:
        seed = self.rng.randrange(self.seed_space)
        header = _int_to_bits(seed, self.header_bits)
        subset = _subset_from_seed(
            seed, self.l, self.degree_cdf, singleton_limit=self.k
        )
        payload = self._xor_subset(subset)
        if self.header_bits == 0:
            return payload.astype(np.uint8)
        if self.symbol_bits == 0:
            return header
        return np.concatenate([header, payload]).astype(np.uint8)


class RaptorRecoverer(Recoverer):
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self.payload_bits = _payload_bits_for_d(d)

        layout = _select_layout(n, self.payload_bits)
        self.header_bits = layout.header_bits
        self.symbol_bits = layout.symbol_bits
        self.k = layout.k
        self.s = layout.s
        self.h = layout.h
        self.l = layout.intermediate_symbols

        self.degree_cdf = _robust_soliton_cdf(self.l)
        self.symbols: list[np.ndarray | None] = [None for _ in range(self.l)]
        self.pending: list[tuple[set[int], np.ndarray]] = []
        self.recovered_index: int | None = 0 if self.payload_bits == 0 else None
        self.precode_graph = _build_precode_graph(n, self.symbol_bits, layout)
        self._inject_precode_constraints()

    def _inject_precode_constraints(self):
        if self.symbol_bits == 0:
            zero = np.zeros(0, dtype=np.uint8)
        else:
            zero = np.zeros(self.symbol_bits, dtype=np.uint8)
        for idx, neighbors in self.precode_graph.items():
            subset = set(neighbors)
            subset.add(idx)
            self.pending.append((subset, zero.copy()))
        self._propagate_pending()

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
                    target = next(iter(subset))
                    if self.symbols[target] is None:
                        self.symbols[target] = payload.copy()
                        progress = True
                    continue

                new_pending.append((subset, payload))

            self.pending = new_pending
            if progress:
                continue
            self._attempt_dense_recovery()

    def _try_finalize(self) -> int | None:
        if self.recovered_index is not None:
            return self.recovered_index
        if any(self.symbols[i] is None for i in range(self.k)):
            return None
        symbols = [self.symbols[i] for i in range(self.k)]
        self.recovered_index = _symbols_to_index(symbols) if symbols else 0
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
            seed, self.l, self.degree_cdf, singleton_limit=self.k
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

    def _attempt_dense_recovery(self):
        unresolved = [i for i in range(self.l) if self.symbols[i] is None]
        if not unresolved or not self.pending:
            return

        index_map = {idx: pos for pos, idx in enumerate(unresolved)}
        filtered: list[tuple[list[int], np.ndarray]] = []
        for subset, payload in self.pending:
            subset_list = [idx for idx in subset if idx in index_map]
            if not subset_list:
                continue
            filtered.append((subset_list, payload.copy()))

        if len(filtered) < len(unresolved):
            return

        cols = len(unresolved)
        rows = len(filtered)
        A = np.zeros((rows, cols), dtype=np.uint8)
        B = np.zeros((rows, self.symbol_bits), dtype=np.uint8)

        for r, (subset_list, payload) in enumerate(filtered):
            for idx in subset_list:
                A[r, index_map[idx]] ^= 1
            B[r] = payload

        solution = self._solve_dense_system(A, B)
        if solution is None:
            return

        for pos, idx in enumerate(unresolved):
            self.symbols[idx] = solution[pos].copy()
        self.pending.clear()

    @staticmethod
    def _solve_dense_system(A: np.ndarray, B: np.ndarray) -> np.ndarray | None:
        rows, cols = A.shape
        if cols == 0:
            return np.zeros((0, B.shape[1]), dtype=np.uint8)
        aug = np.concatenate([A.copy(), B.copy()], axis=1)
        rhs_cols = B.shape[1]

        pivot_row = 0
        pivot_cols: list[int] = []

        for col in range(cols):
            pivot = None
            for r in range(pivot_row, rows):
                if aug[r, col]:
                    pivot = r
                    break
            if pivot is None:
                continue
            if pivot != pivot_row:
                aug[[pivot_row, pivot]] = aug[[pivot, pivot_row]]
            for r in range(rows):
                if r != pivot_row and aug[r, col]:
                    aug[r] ^= aug[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row == cols:
                break

        for r in range(pivot_row, rows):
            if not np.any(aug[r, :cols]) and np.any(aug[r, cols:]):
                return None

        if pivot_row < cols:
            return None

        solution = np.zeros((cols, rhs_cols), dtype=np.uint8)
        for r, col in enumerate(pivot_cols):
            solution[col] = aug[r, cols:]
        return solution


def override_D(n: int) -> int:
    max_bits = _max_payload_bits_for_n(n)
    return 1 << max_bits if max_bits > 0 else 1


def producer_constructor(index: int, n: int, d: int) -> Producer:
    payload_bits = _payload_bits_for_d(d)
    max_bits = _max_payload_bits_for_n(n)
    if payload_bits > max_bits:
        raise ValueError(
            f"Cannot encode payload with {payload_bits} bits, max supported for n={n} is {max_bits}"
        )
    return RaptorProducer(n, index, d)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    payload_bits = _payload_bits_for_d(d)
    max_bits = _max_payload_bits_for_n(n)
    if payload_bits > max_bits:
        raise ValueError(
            f"Cannot recover payload with {payload_bits} bits, max supported for n={n} is {max_bits}"
        )
    return RaptorRecoverer(n, d)
