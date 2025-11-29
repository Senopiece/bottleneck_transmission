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
#   payload_index  ->  Y (nonzero output columns)
#                    -> A = Y X^{-1}
#
# where X is built from F(x_i) and is invertible over GF(2).
# This mapping restricts us to "good" matrices A that are
# guaranteed recoverable under the traversal protocol.
#
# The maximal *guaranteed*-recoverable payload space for given
# (n, m) is of size:
#       D = (2^n - 1)^m
#
# (not all 2^(n*m) matrices are safely recoverable without an
# explicit probe header).


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
    # convert to int mask
    v = vector_to_int(v_bits)

    # eliminate using existing basis rows
    for r in basis:
        pivot = 1 << (r.bit_length() - 1)
        if v & pivot:
            v ^= r

    if v == 0:
        return False, basis

    # Now v is new pivot
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

    Monomial masks are integers 0..(2^N - 1) where:
      - bit i set in the mask means x_i participates
      - mask 0 corresponds to the constant '1' monomial
    """
    masks = list(range(1 << N))  # include s=0 (constant term)
    masks.sort(key=lambda s: (bin(s).count("1"), s))
    return masks[:M]


def anf_feature_vector(x_bits, monomial_masks):
    """
    Compute F(x): the ANF monomial feature vector.
    x_bits: shape (N,), dtype uint8
    monomial_masks: list[int] of length M
    """
    # convert x to mask
    xmask = 0
    for b in x_bits:
        xmask = (xmask << 1) | int(b)

    out = []
    for m in monomial_masks:
        # m == 0 is the constant monomial (always 1)
        if m == 0:
            out.append(1)
        else:
            out.append(1 if (xmask & m) == m else 0)
    return np.array(out, dtype=np.uint8)


# ================================================================
# ========  NEW SECTION — Basis construction for mapping  ========
# ================================================================


def build_feature_basis(n, m, masks):
    """
    Construct a basis of size m from x ∈ {0,1,...,2^n−1}
    such that F(x_i) are linearly independent.
    Returns:
        x_basis  : list of m integers in [0, 2^n−1]
        X        : m×m matrix whose columns are F(x_i)
        Xinv     : its inverse mod2

    With the constant term included, F(0) is nonzero and may
    participate in the basis.
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
        raise ValueError("Could not build feature basis for mapping (with const).")

    X = np.column_stack(Fcols)
    # invert X over GF(2)
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

    We use a simple bijection:
        digit d in [0, 2^n-2]  ->  value = d+1 in [1, 2^n-1]
        and interpret 'value' as an n-bit vector.
    """
    q = (1 << n) - 1  # number of nonzero n-bit vectors
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
    Each column must be nonzero (by construction for our codebook).
    """
    q = (1 << n) - 1
    index = 0
    base = 1
    for i in range(m):
        col = Y[:, i]
        val = 0
        for b in col:
            val = (val << 1) | int(b)
        if val == 0:
            raise ValueError("Decode error: encountered zero column in Y.")
        d = val - 1
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
):
    """
    Raw-matrix nonlinear producer implementing the same traversal logic
    as the fullrank version with tails, but using an ANF feature map
    that INCLUDES the constant term.

    States evolve as:  x_{t+1} = A F(x_t)
    where F(x) is the ANF monomial feature map (with constant).

    Traversal pseudocode implemented exactly:

        tails = states with no incoming transitions
        visited = {0}
        prev = 0
        curr = 1

        while True:
            if curr in visited:
                if curr != 0 and (allow_consecutive_repetitions or prev != 0):
                    yield curr
                yield 0
                if all states visited: visited = {0}
                prev = 0
                if tails - visited nonempty:
                    curr = random choice from tails - visited
                else:
                    curr = random choice from all - visited

            yield curr
            visited.add(curr)
            prev = curr
            curr = A@F(curr)

    NOTE: MAPPING ADDED
    -------------------
    Previously (in the raw version), A was obtained by unpacking
    n*m bits directly from 'index'.

    Here, we interpret 'payload_index' as an element of a smaller
    but guaranteed-recoverable space of size D = (2^n − 1)^m:
        payload_index -> Y -> A = Y X^{-1},
    where columns of Y are nonzero and X is a basis of feature
    vectors F(x_i).
    """

    # The caller should use d = (2^n - 1)^m for this scheme.
    q = (1 << n) - 1
    if d != q**m:
        raise ValueError(
            f"Expected d = (2^n - 1)^m = {q}^{m} for guaranteed-recoverable mapping."
        )

    # 1. Feature masks with constant term
    masks = generate_monomial_masks_with_const(n, m)

    # 2. Build feature basis and its inverse transform matrix X^{-1}
    x_basis, X, Xinv = build_feature_basis(n, m, masks)

    # 3. Map payload index to Y, then to A = Y X^{-1}
    Y = encode_payload_index_to_Y(payload_index, n, m)
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

    # ------------------------------------------------------------
    # 3. Compute TAILS: states with no incoming transitions
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
    # 4. Traversal state (same pseudocode)
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

        # Case 1: revisiting a state
        if curr in visited:

            # Optionally output the repeated state
            if curr != 0 and (allow_consecutive_repetitions or prev != 0):
                if verbose:
                    print(f"[Producer/const] revisit → yield {curr:0{n}b}")
                yield int_to_vec(curr)

            # Always output separator
            if verbose:
                print("[Producer/const] yield separator 0…0")
            yield np.zeros(n, dtype=np.uint8)

            # Reset visited if full coverage
            if len(visited) == Nstates:
                if verbose:
                    print("[Producer/const] visited full → reset to {0}")
                visited = {0}

            # Clear prev
            prev = 0

            # Choose next starting state
            tails_unvisited = set(tails) - visited
            remaining = all_states - visited

            if tails_unvisited:
                curr = random.choice(list(tails_unvisited))
                if verbose:
                    print(f"[Producer/const] jump to TAIL {curr:0{n}b}")
            else:
                curr = random.choice(list(remaining))
                if verbose:
                    print(f"[Producer/const] jump to ANY unvisited {curr:0{n}b}")

            # Continue — do NOT emit curr here

        # Case 2: normal new state
        if verbose:
            print(f"[Producer/const] yield {curr:0{n}b}")

        yield int_to_vec(curr)

        # Mark visited
        visited.add(curr)
        prev = curr

        # Nonlinear update
        curr = nonlinear_step(curr)


# ================================================================
# ===================  Nonlinear Recoverer  ======================
# ================================================================


class NonlinearRecovererWithConst(Recoverer):
    """Recover payload_index from transitions (x → y = A F(x)) with constant-term ANF."""

    def __init__(self, n, m, verbose=False):
        self.n = n
        self.m = m
        self.verbose = verbose

        self.prev = None
        self.transitions = set()
        self.recovered_payload_index = None

        # feature map with constant term
        self.masks = generate_monomial_masks_with_const(n, m)

        # same basis as in producer side (conceptually)
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
                print("[Recoverer/const] SEPARATOR (zero vector) → reset prev")
            self.prev = None
            return self.recovered_payload_index

        # If we have a previous nonzero, record transition
        if self.prev is not None and not np.all(self.prev == 0):
            pair = (tuple(self.prev.tolist()), tuple(data.tolist()))
            if pair not in self.transitions:
                if self.verbose:
                    print(
                        f"[Recoverer/const] ADD transition: {self.prev.tolist()} → {data.tolist()}"
                    )
                self.transitions.add(pair)
            else:
                if self.verbose:
                    print("[Recoverer/const] Transition duplicate, ignoring")
        self.prev = data.copy()

        # If already recovered
        if self.recovered_payload_index is not None:
            return self.recovered_payload_index

        # Need at least m transitions to hope for M×M matrix
        if len(self.transitions) < self.m:
            if self.verbose:
                print(
                    f"[Recoverer/const] Not enough transitions yet ({len(self.transitions)}/{self.m})"
                )
            return None

        # ======================================================
        # Collect m independent F(x) vectors
        # ======================================================
        basis = []
        UX = []
        UY = []
        count = 0

        if self.verbose:
            print(
                f"\n[Recoverer/const] BEGIN collecting independent F(x); transitions={len(self.transitions)}"
            )

        for x_bits, y_bits in self.transitions:
            x = np.array(x_bits, dtype=np.uint8)
            Fx = anf_feature_vector(x, self.masks)
            y = np.array(y_bits, dtype=np.uint8)

            if self.verbose:
                print(f"  F(x)={Fx.tolist()}  from x={x.tolist()}")

            indep, new_basis = add_to_basis_rref_vector(Fx, basis)

            if indep:
                if self.verbose:
                    print(
                        f"    → INDEPENDENT (pivot bit={(new_basis[-1].bit_length()-1)})"
                    )
                    print(f"      basis size grows {len(basis)} → {len(new_basis)}")

                basis = new_basis
                UX.append(Fx)
                UY.append(y)
                count += 1
            else:
                if self.verbose:
                    print("    → DEPENDENT (ignored)")

            if count == self.m:
                break

        if count < self.m:
            if self.verbose:
                print(
                    f"[Recoverer/const] Only found {count}/{self.m} independent F(x) — cannot solve yet."
                )
            return None

        X_obs = np.column_stack(UX)
        Y_obs = np.column_stack(UY)

        if self.verbose:
            print(f"\n[Recoverer/const] BUILT MATRICES:")
            print(f"  X_obs shape = {X_obs.shape}")
            print(f"  Y_obs shape = {Y_obs.shape}")
            print(f"  X_obs=\n{X_obs}")
            print(f"  Y_obs=\n{Y_obs}")

        # Solve A = Y_obs X_obs^{-1} over GF(2)
        try:
            A = self._solve_mod2(Y_obs, X_obs)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("[Recoverer/const] X_obs was singular → retry later")
            return None

        if self.verbose:
            print("[Recoverer/const] SUCCESS — solved A:")
            print(A)

        # Map A back to payload index via A X = Y
        Y_rec = (A @ self.X) % 2
        payload_index = decode_Y_to_payload_index(Y_rec, self.n, self.m)

        if self.verbose:
            print("[Recoverer/const] Y_rec = A X =\n", Y_rec)
            print("[Recoverer/const] payload_index =", payload_index)

        self.recovered_payload_index = payload_index
        return payload_index

    # --------------------------------------
    @staticmethod
    def _solve_mod2(Y, X):
        M = X.shape[0]  # square
        aug = np.concatenate([X.copy(), np.eye(M, dtype=np.uint8)], axis=1)
        # Gauss–Jordan eliminate X → I_M
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
        return (Y @ Xinv) % 2  # N×M


# ================================================================
# ========  D, producer_constructor, recoverer_constructor  ======
# ================================================================


def override_D_with_const(n, m):
    """
    Theoretically, with constant-term ANF and dimension m, one could
    define payloads over 2^(n*m) different matrices.

    However, without an explicit probe header, only a subset of those
    matrices are guaranteed recoverable under the traversal protocol.

    In this implementation we use the maximal guaranteed-safe payload
    space:
        D = (2^n - 1)^m
    corresponding exactly to all choices of nonzero output columns Y_i.
    """
    q = (1 << n) - 1
    return q**m


def _infer_m_from_d_with_const(n, d):
    """
    Given n and payload space size d = (2^n - 1)^m, solve for m.

    This matches the mapping layer that ensures all payloads are
    recoverable.
    """
    q = (1 << n) - 1
    m = 0
    prod = 1
    while prod < d:
        prod *= q
        m += 1
    if prod != d:
        raise ValueError(f"d={d} is not (2^n - 1)^m for integer m")
    if m > (1 << n):  # with constant term, max monomials M = 2^n
        raise ValueError("m > 2^n is impossible for ANF monomials with constant term")
    return m


def override_D(n):
    """
    Note: the maximal guaranteed-recoverable payload size is:
        D = (2^n - 1)^m
    for m in [1, 2^n)
    TODO: hovewer in comparison to l version this appears to have some fails for high m, investigate is it degeneracy or just requires big sequences
    """
    return ((1 << n) - 1) ** 13


def producer_constructor(index: int, n: int, d: int) -> Producer:
    """
    Automatically choose M such that d = (2^n - 1)^M and M ≤ 2^n,
    then use the constant-term ANF mapping with guaranteed-recoverable
    matrices.
    """
    m = _infer_m_from_d_with_const(n, d)
    return GeneratorProducer(nonlinear_producer_with_const(n, m, index, d))


def recoverer_constructor(n: int, d: int) -> Recoverer:
    """
    Recoverer counterpart for the constant-term ANF variant.
    """
    m = _infer_m_from_d_with_const(n, d)
    return NonlinearRecovererWithConst(n, m)


# TODO: also account for that y = AF(x) is not allowed for y = x, make it optional
# TODO: maybe not to remove constant term, but restrict A to be such that only AF(x) = x works only for x = 0, and no other x is mapping to 0 also (so no AF(x) = 0 for any nonzero x)
# TODO: write full fledged readme explaining why and how index map is derived under these constraints, understand it by myself
# TODO: make a simple mapper from bytes to matrix A, maybe adding some constant block Z, such that payload is encoded in integer amount of bits
