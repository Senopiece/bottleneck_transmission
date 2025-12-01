import numpy as np
import random

from ._interface import Producer, Recoverer, GeneratorProducer

# Works by generating transitions of the form:
#     x = A@F(x)
# Possible questions:
# - Why ANF? - to make rank([F(0), F(1) ... F(2^n-1)]) = M
# - Why to make rank([...]) = M? - coz we would need A = Y@F(X)^-1
#
# This is a variant where the ANF monomial feature map
#     *includes* the constant term.
#
# IMPORTANT NOTE:
# ----------------
# A mapping layer payload_index <-> A was added:
#
#   payload_index  ->  Y (good output columns)
#                    -> A = Y X^{-1}
#
# where X is built from F(x_i) and is invertible over GF(2).
#
# NEW: We now also ensure that transitions of form
#       (0,0), (x,0), (0,x), (x,x)
#     will not appear for the basis states and (optionally)
#     for any state, making A fully recoverable under censorship.
#
# This mapping restricts us to matrices A that remain
# guaranteed-recoverable even if those transitions are erased.
#
# The maximal safe payload space used here is:
#       D = (2^n - 2)^m
#
# Why (2^n - 2)? Because for each basis input x_i we forbid
#    y_i = 0    (would be censored)
#    y_i = x_i  (would be censored)
# so each column Y_i has exactly (2^n - 2) possible values.
#
# (Much larger than the "always working" subset for raw mapping,
#  and guaranteed to survive censoring.)


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


def generate_monomial_masks_with_const(N, M):
    """
    Return the first M ANF monomials in degree-then-lex order,
    INCLUDING the constant monomial.
    """
    masks = list(range(1 << N))  # include s=0
    masks.sort(key=lambda s: (bin(s).count("1"), s))
    return masks[:M]


def anf_feature_vector(x_bits, monomial_masks):
    """
    Compute F(x): the ANF monomial feature vector.
    """
    xmask = 0
    for b in x_bits:
        xmask = (xmask << 1) | int(b)

    out = []
    for m in monomial_masks:
        if m == 0:
            out.append(1)
        else:
            out.append(1 if (xmask & m) == m else 0)
    return np.array(out, dtype=np.uint8)


# ================================================================
# ========  Basis construction for mapping (with const)  =========
# ================================================================


def build_feature_basis(n, m, masks):
    """
    Construct a basis of size m from states 0..2^n-1 such that F(x_i)
    are linearly independent. Allows F(0) since constant term included.
    Returns:
        x_basis  : list of m states
        X        : m×m matrix with F(x_i) columns
        Xinv     : inverse over GF(2)
    """
    basis = []
    x_basis = []
    Fcols = []

    for x in range(0, 1 << n):
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

    # invert X mod2
    M = X.shape[0]
    aug = np.concatenate([X.copy(), np.eye(M, dtype=np.uint8)], axis=1)
    for i in range(M):
        pivot = None
        for r in range(i, M):
            if aug[r, i]:
                pivot = r
                break
        if pivot is None:
            raise ValueError("X singular.")
        if pivot != i:
            aug[[i, pivot]] = aug[[pivot, i]]
        for r in range(M):
            if r != i and aug[r, i]:
                aug[r] ^= aug[i]

    Xinv = aug[:, M:]
    return x_basis, X, Xinv


# ================================================================
# ======== Special codebook for censorship-safe mapping  =========
# ================================================================


def allowed_outputs_for_basis_state(x_int, n):
    """
    For a fixed basis input x, allowed outputs y are:
        y != 0
        y != x
    That is 2^n - 2 values.
    We return them as a list of integers in sorted order.
    """
    all_vals = list(range(1, 1 << n))  # nonzero outputs
    # remove the one equal to x
    return [v for v in all_vals if v != x_int]


def encode_payload_index_to_Y_censored(index, n, m, x_basis):
    """
    Encode payload index (0 ≤ index < (2^n−2)^m) into Y columns,
    each chosen from the allowed outputs for its basis state x_i.

    Digitization is in base q = (2^n - 2).
    """
    q = (1 << n) - 2
    cols = []
    for i in range(m):
        d = index % q
        index //= q

        x_i = x_basis[i]
        allowed = allowed_outputs_for_basis_state(x_i, n)

        val = allowed[d]  # choose the d-th allowed output

        col = np.array([(val >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.uint8)
        cols.append(col)

    return np.column_stack(cols)


def decode_Y_to_payload_index_censored(Y, n, m, x_basis):
    """
    Reverse mapping of encode_payload_index_to_Y_censored.
    """
    q = (1 << n) - 2
    index = 0
    base = 1

    for i in range(m):
        col = Y[:, i]
        val = 0
        for b in col:
            val = (val << 1) | int(b)

        x_i = x_basis[i]
        allowed = allowed_outputs_for_basis_state(x_i, n)

        # find d such that allowed[d] == val
        d = allowed.index(val)

        index += d * base
        base *= q

    return index


# ================================================================
# ==================  Nonlinear Producer  =========================
# ================================================================


def nonlinear_producer_with_const(
    n: int,
    m: int,
    payload_index: int,
    d: int,
    allow_consecutive_repetitions: bool = False,
    verbose: bool = False,
    require_global_no_fixed_points: bool = False,  # new option
):
    """
    ANF nonlinear producer (with constant term feature map) using
    censorship-safe index<->A mapping.

    Ensures transitions (0,0), (x,0), (0,x), (x,x) do not appear
    for the basis states, so recoverer survives censorship.

    Optionally enforces global A F(x) != 0 and != x for all states x
    (slower but ultra-safe).
    """

    # d must equal (2^n - 2)^m
    q = (1 << n) - 2
    if d != q**m:
        raise ValueError(f"Expected d = (2^n - 2)^m = {q}^{m}, got d={d}.")

    # 1. Feature masks with constant term
    masks = generate_monomial_masks_with_const(n, m)

    # 2. Build feature basis X and its inverse
    x_basis, X, Xinv = build_feature_basis(n, m, masks)

    # ------------------------------------------------------------
    # 3. Map payload index -> Y (censorship-safe) -> A = Y X^{-1}
    # ------------------------------------------------------------

    Y = encode_payload_index_to_Y_censored(payload_index, n, m, x_basis)
    A = (Y @ Xinv) % 2

    # Optional stronger safety: remove global fixed points
    if require_global_no_fixed_points:

        def violates_global(A):
            for x in range(1 << n):
                xb = np.array(
                    [(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8
                )
                Fx = anf_feature_vector(xb, masks)
                yb = (A @ Fx) % 2
                y = 0
                for b in yb:
                    y = (y << 1) | int(b)
                if y == 0 or y == x:
                    return True
            return False

        # increment payload_index until a valid A appears
        original = payload_index
        while violates_global(A):
            payload_index = (payload_index + 1) % d
            Y = encode_payload_index_to_Y_censored(payload_index, n, m, x_basis)
            A = (Y @ Xinv) % 2

    if verbose:
        print("[Producer/const] x_basis =", x_basis)
        print("[Producer/const] X =\n", X)
        print("[Producer/const] Y =\n", Y)
        print("[Producer/const] A = Y X^{-1} =\n", A)

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def int_to_vec(x):
        return np.array([(x >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.uint8)

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

    # ------------------------------------------------------------
    # 3. Compute TAILS: states with no incoming transitions
    # TODO: for large n optimize this
    # ------------------------------------------------------------
    Nstates = 1 << n
    incoming_count = [0] * Nstates
    for x in range(Nstates):
        y = nonlinear_step(x)
        incoming_count[y] += 1

    tails = [x for x in range(Nstates) if incoming_count[x] == 0]

    if verbose:
        print(f"[Producer/const] Tails: {tails}")

    # ------------------------------------------------------------
    # Traversal logic (unchanged)
    # ------------------------------------------------------------
    visited = {0}
    all_states = set(range(Nstates))
    prev = 0
    curr = 1

    if verbose:
        print("[Producer/const] Start traversal at curr=1")

    # ------------------------------------------------------------
    # 5. Main loop
    # ------------------------------------------------------------
    while True:

        if curr in visited:

            if curr != 0 and (allow_consecutive_repetitions or prev != 0):
                if verbose:
                    print(f"[Producer/const] revisit → yield {curr:0{n}b}")
                yield int_to_vec(curr)

            if verbose:
                print("[Producer/const] yield separator 0…0")
            yield np.zeros(n, dtype=np.uint8)

            if len(visited) == Nstates:
                if verbose:
                    print("[Producer/const] visited full → reset to {0}")
                visited = {0}

            prev = 0

            tails_unvisited = set(tails) - visited
            remaining = all_states - visited

            if tails_unvisited:
                curr = random.choice(list(tails_unvisited))
            else:
                curr = random.choice(list(remaining))

        if verbose:
            print(f"[Producer/const] yield {curr:0{n}b}")

        yield int_to_vec(curr)
        visited.add(curr)
        prev = curr
        curr = nonlinear_step(curr)


# ================================================================
# ===================  Nonlinear Recoverer  ======================
# ================================================================


class NonlinearRecovererWithConst(Recoverer):
    """Recover censorship-safe payload_index from transitions."""

    def __init__(self, n, m, verbose=False):
        self.n = n
        self.m = m
        self.verbose = verbose

        self.prev = None
        self.transitions = set()
        self.recovered_payload_index = None

        self.masks = generate_monomial_masks_with_const(n, m)

        # same basis as producer
        self.x_basis, self.X, self.Xinv = build_feature_basis(n, m, self.masks)

    def feed(self, data):
        if data is None:
            if self.verbose:
                print("[Recoverer/const] GAP → reset prev")
            self.prev = None
            return self.recovered_payload_index

        data = np.array(data, dtype=np.uint8) % 2

        # Zero separator
        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer/const] SEPARATOR → reset prev")
            self.prev = None
            return self.recovered_payload_index

        # record transition if previous nonzero
        if self.prev is not None and not np.all(self.prev == 0):
            pair = (tuple(self.prev.tolist()), tuple(data.tolist()))
            if pair not in self.transitions:
                self.transitions.add(pair)
        self.prev = data.copy()

        if self.recovered_payload_index is not None:
            return self.recovered_payload_index

        if len(self.transitions) < self.m:
            return None

        # collect m independent transitions
        basis = []
        UX = []
        UY = []
        for x_bits, y_bits in self.transitions:
            x = np.array(x_bits, dtype=np.uint8)
            Fx = anf_feature_vector(x, self.masks)
            y = np.array(y_bits, dtype=np.uint8)

            indep, new_basis = add_to_basis_rref_vector(Fx, basis)
            if indep:
                basis = new_basis
                UX.append(Fx)
                UY.append(y)
                if len(UX) == self.m:
                    break

        if len(UX) < self.m:
            return None

        X_obs = np.column_stack(UX)
        Y_obs = np.column_stack(UY)

        try:
            A = self._solve_mod2(Y_obs, X_obs)
        except np.linalg.LinAlgError:
            return None

        # Recover Y = A X
        Y_rec = (A @ self.X) % 2
        payload_index = decode_Y_to_payload_index_censored(
            Y_rec, self.n, self.m, self.x_basis
        )

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
    Safe payload size with censorship:
        D = (2^n - 2)^m
    m up to 2^n
    """
    return ((1 << n) - 2) ** 13


def _infer_m_from_d_censoring(n, d):
    q = (1 << n) - 2
    m = 0
    prod = 1
    while prod < d:
        prod *= q
        m += 1
    if prod != d:
        raise ValueError(f"d={d} not equal (2^n - 2)^m")
    if m > (1 << n):
        raise ValueError("m > 2^n impossible for ANF monomials with constant term")
    return m


def producer_constructor(index: int, n: int, d: int) -> Producer:
    m = _infer_m_from_d_censoring(n, d)
    return GeneratorProducer(nonlinear_producer_with_const(n, m, index, d))


def recoverer_constructor(n: int, d: int) -> Recoverer:
    m = _infer_m_from_d_censoring(n, d)
    return NonlinearRecovererWithConst(n, m)


# TODO: adopt to large packet sizes:
#  - make semi-optimal but computationaly easy index map with smth like extended byte stuffing
#  - generatoreven with this generation strategy, generator also needs to be optimized. but firther research other generation strategies at all
#  - so split onto two impls: one for transferring numbers (full (2^n-2)^m coverage), small packets like now, still optimize a little, other for transferring bytearrays, so the other is constructed from some given payload size in bytes, aligning it somehow, packing into extended byte stuffing for generating good action matrix and using it. the second option is with generator(payload, payload_size, packet_size) -> series, recoverer(payload_size, packet_size) -> bytes
