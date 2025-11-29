import numpy as np
import random

from ._interface import Producer, Recoverer, GeneratorProducer

# ================================================================
#
#      anfext2l mapped the payload index → raw matrix A
#      by simple unpacking of n*m bits.
#
#      That created matrices A for which the recoverer could NOT
#      reconstruct the payload (because the set
#           {F(x) : A F(x) != 0}
#      did not span dimension m).
#
#      A NEW MAPPING WAS ADDED:
#
#        payload_index <--> GOOD MATRIX A
#
#      where "good" matrices are exactly those that remain
#      recoverable under the traversal protocol.
#
#      The mapping uses:
#          A = Y X^{-1}
#      where X is a fixed m×m invertible matrix of feature vectors
#      F(x_i), and Y is composed of m independent nonzero n-bit
#      columns encoded from the payload index in base-(2^n−1).
#
#      Recoverer now returns the payload index, NOT the raw matrix
#      index.
#
# ================================================================

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
# ===============  ANF monomials and feature map  ================
# ================================================================


def generate_monomial_masks(N, M):
    """
    Return the first M *non-constant* ANF monomials (degree >= 1)
    in degree-then-lex order.
    """
    masks = [s for s in range(1, 1 << N)]  # skip s=0
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
# ========  NEW SECTION — Basis construction for mapping  ========
# ================================================================

# >>> ADDED <<<


def build_feature_basis(n, m, masks):
    """
    Construct a basis of size m from nonzero x ∈ {1,2,...,2^n−1}
    such that F(x_i) are linearly independent.
    Returns:
        x_basis  : list of m integers in [1, 2^n−1]
        X        : m×m matrix whose columns are F(x_i)
        Xinv     : its inverse mod2
    """
    basis = []
    x_basis = []
    Fcols = []

    for x in range(1, 1 << n):  # skip x=0 because F(0)=0
        xb = np.array([(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)
        Fx = anf_feature_vector(xb, masks)
        indep, new_basis = add_to_basis_rref_vector(Fx, basis)
        if indep:
            basis = new_basis
            x_basis.append(x)
            Fcols.append(Fx)
            if len(basis) == m:
                break

    if len(x_basis) != m:
        raise ValueError("Could not build feature basis for mapping.")

    X = np.column_stack(Fcols)
    # invert X
    M = X.shape[0]
    aug = np.concatenate([X.copy(), np.eye(M, dtype=np.uint8)], axis=1)
    for i in range(M):
        pivot = None
        for r in range(i, M):
            if aug[r, i]:
                pivot = r
                break
        if pivot is None:
            raise ValueError("X is singular during inversion.")
        if pivot != i:
            aug[[i, pivot]] = aug[[pivot, i]]
        for r in range(M):
            if r != i and aug[r, i]:
                aug[r] ^= aug[i]
    Xinv = aug[:, M:]
    return x_basis, X, Xinv


def encode_payload_index_to_Y(index, n, m):
    """
    Convert payload index (0 ≤ index < (2^n−1)^m) into
    Y ∈ GF(2)^{n×m} whose columns are nonzero n-bit vectors.
    """
    q = (1 << n) - 1
    cols = []
    for _ in range(m):
        d = index % q
        index //= q
        val = d + 1
        col = np.array([(val >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)
        cols.append(col)
    return np.column_stack(cols)


def decode_Y_to_payload_index(Y, n, m):
    """
    Convert Y ∈ GF(2)^{n×m} back to payload index.
    Each column must be nonzero.
    """
    q = (1 << n) - 1
    index = 0
    base = 1
    for i in range(m):
        col = Y[:, i]
        val = 0
        for b in col:
            val = (val << 1) | int(b)
        d = val - 1
        index += d * base
        base *= q
    return index


# ================================================================
# ===============  Nonlinear Producer (unchanged logic) ==========
# ================================================================


def nonlinear_producer(
    n: int,
    m: int,
    payload_index: int,
    d: int,
    allow_consecutive_repetitions: bool = False,
    verbose: bool = False,
):
    """
    NOTE: MAPPING ADDED (see top comment in file)

    Previously:
        raw index → raw A by bit-unpacking n*m bits.

    NOW:
        payload_index → Y → A = Y X^{-1}

    This guarantees that all produced matrices A are recoverable,
    because columns of Y correspond to a full-rank set of F(x_i)
    and are forced to be nonzero.
    """
    assert d == ((1 << n) - 1) ** m or d == 2 ** (
        n * m
    ), "d must match payload space (extended mapping)"

    masks = generate_monomial_masks(n, m)

    # >>> ADDED: build basis and mapping
    x_basis, X, Xinv = build_feature_basis(n, m, masks)
    Y = encode_payload_index_to_Y(payload_index, n, m)
    A = (Y @ Xinv) % 2

    if verbose:
        print("[Producer] Mapped payload to A via Y X^{-1}:")
        print("Y=\n", Y)
        print("A=\n", A)

    # ------------------------------------------------------------
    # unchanged remainder of the producer logic
    # ------------------------------------------------------------

    def int_to_vec(x):
        return np.array([(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)

    def vec_to_int(v):
        out = 0
        for b in v:
            out = (out << 1) | int(b)
        return out

    def nonlinear_step(x):
        xb = int_to_vec(x)
        F = anf_feature_vector(xb, masks)
        yb = (A @ F) % 2
        return vec_to_int(yb)

    Nstates = 1 << n
    incoming_count = [0] * Nstates
    for x in range(Nstates):
        y = nonlinear_step(x)
        incoming_count[y] += 1
    tails = [x for x in range(Nstates) if incoming_count[x] == 0]

    visited = {0}
    all_states = set(range(Nstates))
    prev = 0
    curr = 1

    while True:
        if curr in visited:
            if curr != 0 and (allow_consecutive_repetitions or prev != 0):
                yield int_to_vec(curr)

            yield np.zeros(n, dtype=np.uint8)

            if len(visited) == Nstates:
                visited = {0}

            prev = 0
            tails_unvisited = set(tails) - visited
            remaining = all_states - visited

            curr = (
                random.choice(list(tails_unvisited))
                if tails_unvisited
                else random.choice(list(remaining))
            )

        yield int_to_vec(curr)
        visited.add(curr)
        prev = curr
        curr = nonlinear_step(curr)


# ================================================================
# ===================  Nonlinear Recoverer  ======================
# ================================================================


class NonlinearRecoverer(Recoverer):
    """Recover payload_index from transitions (x → y = A F(x))."""

    def __init__(self, n, m, verbose=False):
        self.n = n
        self.m = m
        self.verbose = verbose

        self.prev = None
        self.transitions = set()
        self.recovered_payload_index = None

        self.masks = generate_monomial_masks(n, m)

        # >>> ADDED: basis for decoding
        self.x_basis, self.X, self.Xinv = build_feature_basis(n, m, self.masks)

    def feed(self, data):
        if data is None:
            self.prev = None
            return self.recovered_payload_index

        data = np.array(data, dtype=np.uint8) % 2

        if np.all(data == 0):
            self.prev = None
            return self.recovered_payload_index

        if self.prev is not None and not np.all(self.prev == 0):
            pair = (tuple(self.prev.tolist()), tuple(data.tolist()))
            self.transitions.add(pair)
        self.prev = data.copy()

        if self.recovered_payload_index is not None:
            return self.recovered_payload_index

        if len(self.transitions) < self.m:
            return None

        basis = []
        UX = []
        UY = []
        count = 0

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
            return None

        X = np.column_stack(UX)
        Y = np.column_stack(UY)

        try:
            A = self._solve_mod2(Y, X)
        except:
            return None

        # >>> ADDED: recover payload index from A
        Yrec = (A @ self.X) % 2
        payload_index = decode_Y_to_payload_index(Yrec, self.n, self.m)

        self.recovered_payload_index = payload_index
        return payload_index

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
        return (Y @ Xinv) % 2


# ================================================================
# ========  D, producer_constructor, recoverer_constructor  ======
# ================================================================


def override_D(n):
    """
    Note: the maximal guaranteed-recoverable payload size is:
        D = (2^n - 1)^m
    for m in [1, 2^n)
    """
    return ((1 << n) - 1) ** 13


def _infer_m_from_d(n, d):
    """
    Infer m from payload size D = (2^n - 1)^m.
    """
    q = (1 << n) - 1
    m = 0
    prod = 1
    while prod < d:
        prod *= q
        m += 1
    if prod != d:
        raise ValueError("d is not a power of (2^n - 1)")
    return m


def producer_constructor(index: int, n: int, d: int) -> Producer:
    m = _infer_m_from_d(n, d)
    return GeneratorProducer(nonlinear_producer(n, m, index, d))


def recoverer_constructor(n: int, d: int) -> Recoverer:
    m = _infer_m_from_d(n, d)
    return NonlinearRecoverer(n, m)
