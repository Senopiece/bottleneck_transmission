import random
import numpy as np

from ._interface import Producer, Recoverer
from .single_binary_fullrank import (
    count_fullrank_matrices,
    rank_fullrank_matrix,
    unrank_fullrank_matrix,
)

# NOTE: fails to transmit some degenerate payloads e.g. D = 0


# ---------- Nonlinear family (invertible ANF-based toggles) ----------


def _bit_for_index(n: int, idx: int) -> int:
    """Return bitmask (MSB-first) for coordinate ``idx`` in an n-bit vector."""
    return 1 << (n - 1 - idx)


def anf_toggle_family(n: int) -> list[tuple[int, int]]:
    """
    Enumerate an invertible family of nonlinear maps over GF(2)^n.

    Each element is a pair (target_idx, mask) describing the permutation
        f(x) = x with coordinate target_idx toggled iff all bits in `mask` are 1.

    Constraints for invertibility:
        - mask is non-empty
        - target_idx bit is NOT included in mask (triangular update, involutive)

    Ordering: target_idx ascending, then mask ascending.
    """
    family: list[tuple[int, int]] = []
    for target in range(n):
        target_bit = _bit_for_index(n, target)
        for mask in range(1, 1 << n):
            if mask & target_bit:
                continue
            family.append((target, mask))
    return family


def eval_anf_toggle(
    family: list[tuple[int, int]], idx: int, vec: np.ndarray
) -> np.ndarray:
    """
    Apply the idx-th toggle from `family` to vector `vec` (returns a copy).
    """
    target, mask = family[idx]
    x_int = vector_to_int(vec)
    if (x_int & mask) == mask:
        out = vec.copy()
        out[target] ^= 1
        return out
    return vec.copy()


# ---------- Helper functions ----------


def vector_to_int(bits):
    """Convert a sequence of binary digits into an integer bitmask (MSB-first)."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def add_to_basis_rref(v, basis):
    """Insert a vector into a basis kept in row-reduced echelon style (GF(2))."""
    for r in basis:
        pivot = 1 << (r.bit_length() - 1)
        if v & pivot:
            v ^= r

    if v == 0:
        return False, basis

    pivot = 1 << (v.bit_length() - 1)
    new_basis = []
    for r in basis:
        if r & pivot:
            r ^= v
        new_basis.append(r)
    new_basis.append(v)
    new_basis.sort(reverse=True)
    return True, new_basis


def in_span(v, basis):
    """Check whether vector ``v`` lies in the linear span of ``basis`` over GF(2)."""
    for r in basis:
        pivot = 1 << (r.bit_length() - 1)
        if v & pivot:
            v ^= r
    return v == 0


_DEFAULT_NONLINEAR: dict[int, tuple[list[np.ndarray], list[int]]] = {}


def _default_nonlinear(n: int) -> tuple[list[np.ndarray], list[int]]:
    """
    Generate (and memoize) seven known matrices (B..H) and eight distinct
    nonlinear toggle indices (f0..f7) for size n.
    """
    if n in _DEFAULT_NONLINEAR:
        return _DEFAULT_NONLINEAR[n]

    family = anf_toggle_family(n)
    if len(family) < 8:
        raise ValueError(f"Not enough nonlinear toggles available for n={n}")

    rng = np.random.default_rng()
    f_indices = rng.choice(len(family), size=8, replace=False).astype(int).tolist()
    mats = [rng.integers(0, 2, size=(n, n), dtype=np.uint8) for _ in range(7)]

    _DEFAULT_NONLINEAR[n] = (mats, f_indices)
    return _DEFAULT_NONLINEAR[n]


def _resolve_nonlinear(
    n: int,
    B_list: list[np.ndarray] | None,
    f0_idx: int | None,
    f1_idx: int | None,
    f_indices_full: list[int] | None = None,
) -> tuple[list[tuple[int, int]], list[np.ndarray], list[int]]:
    """Resolve the known nonlinear toggles/matrices for this size (lazy-random defaults)."""
    family = anf_toggle_family(n)
    if B_list is None or f_indices_full is None:
        B_list_def, f_indices_def = _default_nonlinear(n)
    else:
        B_list_def, f_indices_def = B_list, f_indices_full

    if B_list is None:
        B_list = B_list_def
    else:
        B_list = [np.array(M, dtype=np.uint8) for M in B_list]

    if f_indices_full is None:
        if f0_idx is None or f1_idx is None:
            f_indices = f_indices_def
        else:
            f_indices = f_indices_def
    else:
        f_indices = f_indices_full

    assert len(B_list) == 7, "Need seven known matrices for f1..f7"
    assert len(f_indices) == 8, "Need eight toggle indices f0..f7"
    for idx in f_indices:
        assert 0 <= idx < len(family), "f indices must reference the ANF toggle family"
    return family, B_list, f_indices


# ---------- rank / unrank for all matrices ----------


def index_to_matrix(n: int, index: int) -> np.ndarray:
    """Map integer index in [0, 2^(n^2)) to an arbitrary binary n x n matrix."""
    max_val = 1 << (n * n)
    assert 0 <= index < max_val, "Index out of range for n x n matrix payload"

    bits = [(index >> k) & 1 for k in reversed(range(n * n))]
    A = np.array(bits, dtype=np.uint8).reshape((n, n))
    return A


def matrix_to_index(A: np.ndarray) -> int:
    """Inverse of index_to_matrix: encode an n x n binary matrix as an integer."""
    A = np.array(A, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix must be square"

    idx = 0
    for r in range(n):
        for c in range(n):
            idx = (idx << 1) | int(A[r, c] & 1)
    return idx


# ---------- Producer for arbitrary matrices ----------


class MatrixProducer(Producer):
    """
    Producer for arbitrary binary n x n matrices (no full-rank restriction),
    but smart: when starting a new orbit, pick tails first.
    """

    def __init__(
        self,
        n: int,
        index: int,
        B_list: list[np.ndarray] | None = None,
        f0_idx: int | None = None,
        f1_idx: int | None = None,
        f_indices: list[int] | None = None,
        verbose: bool = False,
    ):
        self.n = n
        self.index = index
        self.verbose = verbose

        # Build generator matrix from payload bits
        self.A = index_to_matrix(n, index)
        self.family, self.B_list, self.f_indices = _resolve_nonlinear(
            n, B_list, f0_idx, f1_idx, f_indices_full=f_indices
        )

        # Precompute successor and predecessor graph
        self._build_graph()

        # State
        self.curr: int = 0
        self.visited: set[int] = {0}
        self.all_states: set[int] = set(range(2**n))
        self.unvisited: set[int] = self.all_states - self.visited

        self.after_separator = False
        self.cycle_count = 0

        if self.verbose:
            print(f"[Producer] A=\n{self.A}")
            print(f"[Producer] graph built with {2**n} nodes")

    # ------------------------------------------------------------
    def _int_to_vec(self, x: int) -> np.ndarray:
        return np.array(
            [(x >> (self.n - 1 - b)) & 1 for b in range(self.n)], dtype=np.uint8
        )

    def _vec_to_int(self, v: np.ndarray) -> int:
        return vector_to_int(v)

    # ------------------------------------------------------------
    def _build_graph(self):
        """Build successor and predecessor lists for x -> A f0(x) + f1(x)."""
        self.succ = {}
        self.pred = {i: [] for i in range(2**self.n)}

        for x in range(2**self.n):
            v = self._int_to_vec(x)
            nxt = self._update_vec(v)
            y = self._vec_to_int(nxt)

            self.succ[x] = y
            self.pred[y].append(x)

    def _update_vec(self, vec: np.ndarray) -> np.ndarray:
        f_vals = [eval_anf_toggle(self.family, idx, vec) for idx in self.f_indices]
        acc = (self.A @ f_vals[0]) % 2
        for M, fv in zip(self.B_list, f_vals[1:]):
            acc ^= (M @ fv) % 2
        return acc

    # ------------------------------------------------------------
    def _find_tails(self) -> list[int]:
        """Return list of unvisited nodes with no incoming edges from unvisited nodes."""
        tails = []
        for x in self.unvisited:
            incoming = self.pred[x]
            if not any(p in self.unvisited for p in incoming):
                tails.append(x)
        return tails

    # ------------------------------------------------------------
    def _choose_new_start(self) -> int:
        """Choose next start state: prefer tails, otherwise any unvisited node."""
        tails = self._find_tails()
        if tails:
            return random.choice(tails)
        return random.choice(tuple(self.unvisited))

    # ------------------------------------------------------------
    def generate(self) -> np.ndarray:
        """Emit infinite stream of n-bit packets with zero separators."""
        if self.curr in self.visited:
            if not self.after_separator:
                self.after_separator = True
                if self.verbose:
                    print("[Producer] -> separator")
                return np.zeros(self.n, dtype=np.uint8)

            if not self.unvisited:
                self.cycle_count += 1
                self.visited = {0}
                self.unvisited = self.all_states - self.visited
                if self.verbose:
                    print(f"[Producer] RESET visited (cycle #{self.cycle_count})")

            start = self._choose_new_start()
            self.curr = start

            self.visited.add(start)
            if start in self.unvisited:
                self.unvisited.remove(start)

            vec = self._int_to_vec(start)
            nxt = self._update_vec(vec)
            self.curr = self._vec_to_int(nxt)
            self.after_separator = False

            if self.verbose:
                print(
                    f"[Producer] start orbit @ {start:0{self.n}b} "
                    f"(tails={len(self._find_tails())})"
                )

            return vec.copy()

        self.visited.add(self.curr)
        if self.curr in self.unvisited:
            self.unvisited.remove(self.curr)

        vec = self._int_to_vec(self.curr)
        nxt = self._update_vec(vec)
        self.curr = self._vec_to_int(nxt)
        self.after_separator = False

        if self.verbose:
            print(f"[Producer] cont orbit: {vec}")

        return vec.copy()


# ---------- Recoverer for arbitrary matrices ----------


class MatrixRecoverer(Recoverer):
    """
    Recoverer for arbitrary binary n x n matrices with known nonlinear part.
    """

    def __init__(
        self,
        n: int,
        B_list: list[np.ndarray] | None = None,
        f0_idx: int | None = None,
        f1_idx: int | None = None,
        f_indices: list[int] | None = None,
        verbose: bool = False,
    ):
        self.n = n
        self.verbose = verbose

        self.family, self.B_list, self.f_indices = _resolve_nonlinear(
            n, B_list, f0_idx, f1_idx, f_indices_full=f_indices
        )
        self.prev: np.ndarray | None = None
        self.transitions: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        self.recovered_index: int | None = None

    # ------------------------------------------------------------
    def feed(self, data: np.ndarray | None) -> int | None:
        """Feed a new packet or None (gap). Returns recovered payload index or None."""
        if data is None:
            if self.verbose:
                print("[Recoverer] GAP -> reset prev")
            self.prev = None
            return self.recovered_index

        data = np.array(data, dtype=np.uint8) % 2

        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer] Zero packet -> segment break")
            self.prev = None
            return self.recovered_index

        if self.prev is not None and not np.all(self.prev == 0):
            u_raw = self.prev
            w_raw = data
            f_vals = [
                eval_anf_toggle(self.family, idx, u_raw) for idx in self.f_indices
            ]
            u_eff = f_vals[0]
            correction = np.zeros(self.n, dtype=np.uint8)
            for M, fv in zip(self.B_list, f_vals[1:]):
                correction ^= (M @ fv) % 2
            w_eff = (w_raw ^ correction) % 2
            key = (tuple(u_eff.tolist()), tuple(w_eff.tolist()))
            if key not in self.transitions:
                self.transitions.add(key)
                if self.verbose:
                    print(
                        f"[Recoverer] + transition #{len(self.transitions)}: "
                        f"{''.join(map(str, u_eff))} -> {''.join(map(str, w_eff))}"
                    )
            elif self.verbose:
                print(
                    f"[Recoverer]   (duplicate transition ignored, total={len(self.transitions)})"
                )

        self.prev = data.copy()

        if self.recovered_index is not None:
            return self.recovered_index

        if len(self.transitions) < self.n:
            if self.verbose:
                print(
                    f"[Recoverer] Transitions collected: {len(self.transitions)}/{self.n}"
                )
            return None

        us: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        basis: list[int] = []

        for u_bits, w_bits in self.transitions:
            u = np.array(u_bits, dtype=np.uint8)
            umask = vector_to_int(u)

            indep, basis2 = add_to_basis_rref(umask, basis)
            if indep:
                basis = basis2
                us.append(u)
                ws.append(np.array(w_bits, dtype=np.uint8))

            if len(us) == self.n:
                break

        if len(us) < self.n:
            if self.verbose:
                print(
                    f"[Recoverer] Rank {len(us)}/{self.n} -> waiting for more independent transitions."
                )
            return None

        U = np.column_stack(us) % 2
        W = np.column_stack(ws) % 2

        try:
            A = self._solve_mod2(W, U)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("[Recoverer] Singular transition matrix, retrying later")
            return None

        idx = matrix_to_index(A)
        self.recovered_index = idx
        if self.verbose:
            print(f"[Recoverer] Recovered index={idx}")
            print(f"[Recoverer] A=\n{A}")
        return idx

    # ------------------------------------------------------------
    @staticmethod
    def _solve_mod2(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Solve Y = A X for A over GF(2), assuming X is invertible."""
        n = X.shape[0]
        aug = np.concatenate([X.copy(), np.eye(n, dtype=np.uint8)], axis=1)

        for i in range(n):
            pivot = None
            for r in range(i, n):
                if aug[r, i]:
                    pivot = r
                    break
            if pivot is None:
                raise np.linalg.LinAlgError("Singular matrix")
            if pivot != i:
                aug[[i, pivot]] = aug[[pivot, i]]
            for r in range(n):
                if r != i and aug[r, i]:
                    aug[r, :] ^= aug[i, :]

        X_inv = aug[:, n:]
        return (Y @ X_inv) % 2


# ---------- Helpers for full-rank matrix indexing ----------


def fullrank_index_to_matrix(n: int, index: int) -> np.ndarray:
    """Map payload index -> full-rank binary n x n matrix (columns MSB-first)."""
    cols = unrank_fullrank_matrix(n, index)
    return np.array(
        [[(col >> (n - 1 - r)) & 1 for col in cols] for r in range(n)],
        dtype=np.uint8,
    )


def _is_fullrank(A: np.ndarray) -> bool:
    """Check full-rank over GF(2) via RREF-style basis building."""
    n = A.shape[0]
    basis: list[int] = []
    for j in range(n):
        col_mask = vector_to_int(A[:, j])
        indep, basis = add_to_basis_rref(col_mask, basis)
        if not indep:
            return False
    return True


def matrix_to_fullrank_index(A: np.ndarray) -> int:
    """Inverse mapping: full-rank matrix -> payload index (raises if singular)."""
    A = np.array(A, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix must be square"

    basis: list[int] = []
    cols: list[int] = []
    for j in range(n):
        col_mask = vector_to_int(A[:, j])
        cols.append(col_mask)
        indep, basis = add_to_basis_rref(col_mask, basis)
        if not indep:
            raise ValueError("Matrix is not full-rank over GF(2)")

    return rank_fullrank_matrix(n, cols)


# ---------- Producer (full-rank payload wrapper) ----------


class FullrankMatrixProducer(MatrixProducer):
    """Same generator logic as MatrixProducer, but payload indexes only full-rank A."""

    def __init__(
        self,
        n: int,
        index: int,
        B_list: list[np.ndarray] | None = None,
        f0_idx: int | None = None,
        f1_idx: int | None = None,
        f_indices: list[int] | None = None,
        verbose: bool = False,
    ):
        assert (
            0 <= index < count_fullrank_matrices(n)
        ), "index out of range for full-rank matrices"
        self.payload_index = index

        A = fullrank_index_to_matrix(n, index)
        matrix_bits_index = matrix_to_index(A)

        super().__init__(
            n,
            matrix_bits_index,
            B_list=B_list,
            f0_idx=f0_idx,
            f1_idx=f1_idx,
            f_indices=f_indices,
            verbose=verbose,
        )


# ---------- Recoverer (full-rank payload wrapper) ----------


class FullrankMatrixRecoverer(MatrixRecoverer):
    """MatrixRecoverer that post-processes recovered A to a full-rank payload index."""

    def __init__(
        self,
        n: int,
        B_list: list[np.ndarray] | None = None,
        f0_idx: int | None = None,
        f1_idx: int | None = None,
        f_indices: list[int] | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n,
            B_list=B_list,
            f0_idx=f0_idx,
            f1_idx=f1_idx,
            f_indices=f_indices,
            verbose=verbose,
        )
        self.payload_index: int | None = None

    def feed(self, data: np.ndarray | None) -> int | None:
        bit_index = super().feed(data)

        if self.payload_index is not None:
            return self.payload_index
        if bit_index is None:
            return None

        A = index_to_matrix(self.n, bit_index)
        if not _is_fullrank(A):
            if self.verbose:
                print("[Recoverer] Recovered singular A; waiting for more data")
            self.recovered_index = None
            return None

        self.payload_index = matrix_to_fullrank_index(A)
        return self.payload_index


# ---------- constructors for external benchmark harness ----------


override_D = lambda n: count_fullrank_matrices(n)


_GLOBAL_DEFAULTS: dict[int, tuple[list[np.ndarray], list[int]]] = {}


def _init_global_defaults(n: int) -> tuple[list[np.ndarray], list[int]]:
    if n in _GLOBAL_DEFAULTS:
        return _GLOBAL_DEFAULTS[n]
    family = anf_toggle_family(n)
    rng = np.random.default_rng()
    f_indices = rng.choice(len(family), size=8, replace=False).astype(int).tolist()
    mats = [rng.integers(0, 2, size=(n, n), dtype=np.uint8) for _ in range(7)]
    _GLOBAL_DEFAULTS[n] = (mats, f_indices)
    return _GLOBAL_DEFAULTS[n]


def producer_constructor(index: int, n: int, d: int) -> Producer:
    assert d == count_fullrank_matrices(n)
    mats, f_indices = _init_global_defaults(n)
    return FullrankMatrixProducer(n, index, B_list=mats, f_indices=f_indices)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    assert d == count_fullrank_matrices(n)
    mats, f_indices = _init_global_defaults(n)
    return FullrankMatrixRecoverer(n, B_list=mats, f_indices=f_indices)
