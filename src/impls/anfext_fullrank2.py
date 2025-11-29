import numpy as np
import random

from ._interface import Producer, Recoverer, GeneratorProducer

# NOTE: fails to transmit some degenerate payloads e.g. D = 0


# Better nonlinear_producer compared to anfext_fullrank.py:

# ================================================================
# ======================  Helper functions  ======================
# ================================================================


def vector_to_int(bits):
    """Convert binary vector to integer (MSB first)."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def add_to_basis_rref_vector(v_bits, basis):
    """
    v_bits: np array of shape (M,)
    basis: list of int masks (each an M-bit vector)
    """
    v = vector_to_int(v_bits)

    # eliminate using existing basis rows
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


# ================================================================
# === Full-rank enumeration helpers (generalizing your old code) =
# ================================================================

from itertools import product
from math import prod, log2


def all_vectors(n):
    """All non-zero n-bit vectors as int masks."""
    return [vector_to_int(bits) for bits in product([0, 1], repeat=n) if any(bits)]


def add_to_basis_rref_int(v, basis):
    """RREF basis update for int-encoded vectors (like your old add_to_basis_rref)."""
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


def in_span_int(v, basis):
    """Check membership in span(basis) for int-encoded vectors."""
    for r in basis:
        pivot = 1 << (r.bit_length() - 1)
        if v & pivot:
            v ^= r
    return v == 0


# ---------- square full-rank (n×n) rank/unrank ----------


def unrank_fullrank_square(n, index):
    """
    Your old unrank_fullrank_matrix, but renamed to avoid collision.
    Returns list[int] of length n: columns of an n×n full-rank matrix.
    """
    allv = all_vectors(n)
    basis = []
    cols = []
    for step in range(n):
        valid = [v for v in allv if not in_span_int(v, basis)]
        choice = valid[index % len(valid)]
        index //= len(valid)
        cols.append(choice)
        _, basis = add_to_basis_rref_int(choice, basis)
    return cols


def rank_fullrank_square(n, cols):
    """
    Inverse of unrank_fullrank_square.
    cols: list[int] of length n (full-rank n×n).
    """
    allv = all_vectors(n)
    basis = []
    digits = []
    bases = []

    for v in cols:
        valid = [x for x in allv if not in_span_int(x, basis)]
        pos = valid.index(v)
        digits.append(pos)
        bases.append(len(valid))
        _, basis = add_to_basis_rref_int(v, basis)

    i = digits[-1]
    for k in range(n - 2, -1, -1):
        i = digits[k] + bases[k] * i
    return i


def count_fullrank_square(n):
    """Number of full-rank n×n binary matrices."""
    return prod(2**n - 2**i for i in range(n))


# ---------- rectangular n×m full-row-rank via prefix full-rank ----------
# First n columns form an n×n full-rank matrix, remaining m-n arbitrary


def count_fullrank_nm(n, m):
    """
    Count matrices A ∈ GF(2)^{n×m} whose first n columns form a full-rank n×n block.
    This guarantees full row rank = n.
    """
    if m < n:
        raise ValueError("m must be >= n for full row rank.")
    base = count_fullrank_square(n)
    extra = 2 ** (n * (m - n))
    return base * extra


def unrank_fullrank_nm(n, m, index):
    """
    Unrank index in [0, count_fullrank_nm(n,m)) to A ∈ GF(2)^{n×m}
    with first n columns full-rank and remaining arbitrary.
    """
    if m < n:
        raise ValueError("m must be >= n")

    extra_count = 2 ** (n * (m - n))
    base_count = count_fullrank_square(n)
    total = base_count * extra_count
    if not (0 <= index < total):
        raise ValueError("index out of range for full-rank n×m enumeration")

    high = index // extra_count  # which full-rank n×n prefix
    low = index % extra_count  # which tail columns

    # first n columns: full-rank
    prefix_cols = unrank_fullrank_square(n, high)  # list[int], length n

    # tail columns: arbitrary 2^n possibilities each
    tail_cols = []
    for _ in range(m - n):
        col_val = low % (1 << n)
        low //= 1 << n
        tail_cols.append(col_val)

    cols = prefix_cols + tail_cols  # total m columns

    # convert list[int] columns to n×m matrix A (row-major)
    A = np.zeros((n, m), dtype=np.uint8)
    for j, col in enumerate(cols):
        for r in range(n):
            # bit (n-1-r) of col
            A[r, j] = (col >> (n - 1 - r)) & 1
    return A


def rank_fullrank_nm(n, m, A):
    """
    Rank an A ∈ GF(2)^{n×m} that belongs to our enumeration:
    - first n columns must form a full-rank n×n matrix.
    """
    if m < n:
        raise ValueError("m must be >= n")
    A = np.array(A, dtype=np.uint8)
    if A.shape != (n, m):
        raise ValueError("A must be n×m")

    # extract first n columns as int masks
    prefix_cols = []
    for j in range(n):
        col = 0
        for r in range(n):
            col = (col << 1) | int(A[r, j] & 1)
        prefix_cols.append(col)

    # this will raise if prefix not full-rank
    high = rank_fullrank_square(n, prefix_cols)

    # tail columns → base-2^n integer low
    low = 0
    base = 1
    for j in range(n, m):
        col = 0
        for r in range(n):
            col = (col << 1) | int(A[r, j] & 1)
        low += col * base
        base *= 1 << n

    extra_count = 2 ** (n * (m - n))
    return high * extra_count + low


# ================================================================
# ===============  ANF monomials and feature map  ================
# ================================================================


def generate_monomial_masks(N, M):
    """
    Return the first M monomials (as bitmasks) in degree-then-lex order.
    Total possible = 2^N.
    """
    masks = list(range(1 << N))
    masks.sort(key=lambda s: (bin(s).count("1"), s))
    return masks[:M]


def anf_feature_vector(x_bits, monomial_masks):
    """
    Compute F(x): the ANF monomial feature vector.
    x_bits: shape (N,), dtype uint8
    monomial_masks: list[int] of length M
    """
    xmask = 0
    for b in x_bits:
        xmask = (xmask << 1) | int(b)

    out = []
    for m in monomial_masks:
        out.append(1 if (xmask & m) == m else 0)
    return np.array(out, dtype=np.uint8)


# ================================================================
# ==================  Nonlinear Producer  =========================
# ================================================================


def nonlinear_producer(
    n: int,
    m: int,
    index: int,
    d: int,
    allow_consecutive_repetitions: bool = False,
    verbose: bool = False,
):
    """
    New producer implementing traversal logic:

        tails = states with no incoming transitions
        visited = {0}
        prev = 0
        curr = 1

        while True:
            if curr in visited:
                if curr != 0 and (allow_consecutive_repetitions or prev != 0):
                    yield curr
                yield 0
                if all states visited:
                    visited = {0}
                prev = 0
                if (tails - visited) not empty:
                    curr = random choice from tails - visited
                else:
                    curr = random choice from (all - visited)

            yield curr
            visited.add(curr)
            prev = curr
            curr = AF(curr)

    A is a full-row-rank n×m generator built from index.
    """

    # ------------------------------------------------------------
    # 1. Decode payload index → full-row-rank matrix A (n×m)
    # ------------------------------------------------------------
    assert d == count_fullrank_nm(n, m), "d mismatch"
    A = unrank_fullrank_nm(n, m, index)

    if verbose:
        print("Producer A=\n", A)

    # ------------------------------------------------------------
    # 2. Precompute monomial masks for F(x)
    # ------------------------------------------------------------
    masks = generate_monomial_masks(n, m)

    # helpers
    def int_to_vec(x):
        return np.array([(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)

    def vec_to_int(v):
        out = 0
        for b in v:
            out = (out << 1) | int(b)
        return out

    def nonlinear_step(x):
        """Compute AF(x) and return integer next state."""
        xb = int_to_vec(x)
        F = anf_feature_vector(xb, masks)
        yb = (A @ F) % 2
        return vec_to_int(yb)

    # ------------------------------------------------------------
    # 3. Compute all transitions and detect TAILS (no incoming edges)
    # ------------------------------------------------------------
    Nstates = 1 << n
    incoming_count = [0] * Nstates

    # one-step graph defined by x → AF(x)
    for x in range(Nstates):
        y = nonlinear_step(x)
        incoming_count[y] += 1

    tails = [x for x in range(Nstates) if incoming_count[x] == 0]

    if verbose:
        print(f"[Producer] Tails detected: {tails}")

    # ------------------------------------------------------------
    # 4. Traversal engine (as per your pseudocode)
    # ------------------------------------------------------------
    visited = {0}
    all_states = set(range(Nstates))
    prev = 0
    curr = 1

    if verbose:
        print("[Producer] Starting traversal with curr=1")

    # ------------------------------------------------------------
    # 5. Main infinite generator loop
    # ------------------------------------------------------------
    while True:

        # Case 1: revisit
        if curr in visited:
            # emit curr (unless 0 or suppressed repetition)
            if curr != 0 and (allow_consecutive_repetitions or prev != 0):
                if verbose:
                    print(f"[Producer] revisit → yielding {curr:0{n}b}")
                yield int_to_vec(curr)

            # emit separator (zero)
            if verbose:
                print(f"[Producer] yielding separator 0…0")
            yield np.zeros(n, dtype=np.uint8)

            # reset visited if everything visited
            if len(visited) == Nstates:
                if verbose:
                    print("[Producer] Full coverage reached → resetting visited={0}")
                visited = {0}

            # reset prev
            prev = 0

            # next state selection:
            tails_unvisited = set(tails) - visited
            remaining = all_states - visited

            if tails_unvisited:
                curr = random.choice(list(tails_unvisited))
                if verbose:
                    print(f"[Producer] jumping to TAIL {curr:0{n}b}")
            else:
                curr = random.choice(list(remaining))
                if verbose:
                    print(f"[Producer] jumping to ANY unvisited {curr:0{n}b}")

            # Continue loop — but careful:
            # we DO NOT emit curr here, the loop top will handle it.

        # Case 2: normal visit
        if verbose:
            print(f"[Producer] yielding {curr:0{n}b}")

        yield int_to_vec(curr)
        visited.add(curr)
        prev = curr

        # nonlinear update
        curr = nonlinear_step(curr)


# ================================================================
# ===================  Nonlinear Recoverer  ======================
# ================================================================


class NonlinearRecoverer(Recoverer):
    """Recover full-row-rank A from transitions (x → y = A F(x))."""

    def __init__(self, n, m, verbose=False):
        self.n = n
        self.m = m
        self.verbose = verbose

        self.prev = None
        self.transitions = set()
        self.recovered_index = None

        self.masks = generate_monomial_masks(n, m)

    def feed(self, data):
        if data is None:
            if self.verbose:
                print("[Recoverer] GAP → reset prev")
            self.prev = None
            return self.recovered_index

        data = np.array(data, dtype=np.uint8) % 2

        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer] SEPARATOR (zero vector) → reset prev")
            self.prev = None
            return self.recovered_index

        if self.prev is not None and not np.all(self.prev == 0):
            pair = (tuple(self.prev.tolist()), tuple(data.tolist()))
            if pair not in self.transitions:
                if self.verbose:
                    print(
                        f"[Recoverer] ADD transition: {self.prev.tolist()} → {data.tolist()}"
                    )
                self.transitions.add(pair)
        self.prev = data.copy()

        if self.recovered_index is not None:
            return self.recovered_index

        if len(self.transitions) < self.m:
            if self.verbose:
                print(
                    f"[Recoverer] Not enough transitions yet ({len(self.transitions)}/{self.m})"
                )
            return None

        basis = []
        UX = []
        UY = []
        count = 0

        if self.verbose:
            print(
                f"\n[Recoverer] BEGIN collecting independent F(x); transitions={len(self.transitions)}"
            )

        for x_bits, y_bits in self.transitions:
            x = np.array(x_bits, dtype=np.uint8)
            Fx = anf_feature_vector(x, self.masks)
            y = np.array(y_bits, dtype=np.uint8)

            indep, new_basis = add_to_basis_rref_vector(Fx, basis)

            if indep:
                basis = new_basis
                UX.append(Fx)
                UY.append(y)
                count += 1

            if count == self.m:
                break

        if count < self.m:
            if self.verbose:
                print(
                    f"[Recoverer] Only found {count}/{self.m} independent F(x) — cannot solve yet."
                )
            return None

        X = np.column_stack(UX)  # m×m
        Y = np.column_stack(UY)  # n×m

        if self.verbose:
            print(f"\n[Recoverer] BUILT MATRICES:")
            print(f"  X shape = {X.shape}")
            print(f"  Y shape = {Y.shape}")

        try:
            A = self._solve_mod2(Y, X)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("[Recoverer] X was singular → retry later")
            return None

        # Rank A back to payload index using our full-rank n×m scheme
        idx = rank_fullrank_nm(self.n, self.m, A)
        self.recovered_index = idx
        if self.verbose:
            print("[Recoverer] SUCCESS — solved A and ranked index =", idx)
        return idx

    def _matrix_to_index(self, A):
        return rank_fullrank_nm(self.n, self.m, A)

    @staticmethod
    def _solve_mod2(Y, X):
        M = X.shape[0]
        aug = np.concatenate([X.copy(), np.eye(M, dtype=np.uint8)], axis=1)
        for i in range(M):
            pivot = None
            for r in range(i, M):
                if aug[r, i]:
                    pivot = r
                    break
            if pivot is None:
                raise np.linalg.LinAlgError("singular")
            if pivot != i:
                aug[[i, pivot]] = aug[[pivot, i]]
            for r in range(M):
                if r != i and aug[r, i]:
                    aug[r] ^= aug[i]
        Xinv = aug[:, M:]
        return (Y @ Xinv) % 2  # n×m


# ================================================================
# ========  D, producer_constructor, recoverer_constructor  ======
# ================================================================


def override_D(n: int) -> int:
    m_default = 1 << (n - 1)
    return count_fullrank_nm(n, m_default)


def _infer_m_from_d(n: int, d: int) -> int:
    """
    Given n and d, find m >= n such that d == count_fullrank_nm(n, m).
    """
    max_m = 1 << n  # monomial limit
    for m in range(n, max_m + 1):
        if count_fullrank_nm(n, m) == d:
            return m
    raise ValueError(f"d={d} is not count_fullrank_nm(n, m) for any m in [n, 2^n]")


def producer_constructor(index: int, n: int, d: int) -> Producer:
    """
    Automatically choose M such that d = count_fullrank_nm(n, M),
    and use only full-row-rank A as generators.
    """
    m = _infer_m_from_d(n, d)
    return GeneratorProducer(nonlinear_producer(n, m, index, d))


def recoverer_constructor(n: int, d: int) -> Recoverer:
    m = _infer_m_from_d(n, d)
    return NonlinearRecoverer(n, m)
