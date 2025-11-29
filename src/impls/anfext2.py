import numpy as np
import random

from ._interface import Producer, Recoverer, GeneratorProducer

# NOTE: fails to transmit some degenerate payloads e.g. D = 0


# Works by generating transitions of the form:
#     x = A@F(x)
# Possible questions:
# - Why ANF? - to make rank([F(0), F(1) ... F(2^n-1)]) = M
# - Why to make rank([...]) = M? - coz we would need A = Y@F(X)^-1

# Better nonlinear_producer compared to anfext.py:

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
    # convert x to mask
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
    Raw-matrix nonlinear producer implementing the same traversal logic
    as the fullrank version with tails.

    States evolve as:  x_{t+1} = A F(x_t)
    where F(x) is the ANF monomial feature map.

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
    """

    # ------------------------------------------------------------
    # 1. Decode the raw payload index → A (n×m)
    # ------------------------------------------------------------
    assert d == 2 ** (n * m), "invalid d for raw matrix payload"

    bits = [(index >> k) & 1 for k in reversed(range(n * m))]
    A = np.array(bits, dtype=np.uint8).reshape((n, m))

    if verbose:
        print("Producer A=\n", A)

    # ------------------------------------------------------------
    # 2. Precompute monomial masks
    # ------------------------------------------------------------
    masks = generate_monomial_masks(n, m)

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
        print(f"[Producer] Tails: {tails}")

    # ------------------------------------------------------------
    # 4. Traversal state (your pseudocode)
    # ------------------------------------------------------------
    visited = {0}
    all_states = set(range(Nstates))
    prev = 0
    curr = 1

    if verbose:
        print("[Producer] Start traversal at curr=1")

    # ------------------------------------------------------------
    # 5. Main loop
    # ------------------------------------------------------------
    while True:

        # Case 1: revisiting a state
        if curr in visited:

            # Optionally output the repeated state
            if curr != 0 and (allow_consecutive_repetitions or prev != 0):
                if verbose:
                    print(f"[Producer] revisit → yield {curr:0{n}b}")
                yield int_to_vec(curr)

            # Always output separator
            if verbose:
                print("[Producer] yield separator 0…0")
            yield np.zeros(n, dtype=np.uint8)

            # Reset visited if full coverage
            if len(visited) == Nstates:
                if verbose:
                    print("[Producer] visited full → reset to {0}")
                visited = {0}

            # Clear prev
            prev = 0

            # Choose next starting state
            tails_unvisited = set(tails) - visited
            remaining = all_states - visited

            if tails_unvisited:
                curr = random.choice(list(tails_unvisited))
                if verbose:
                    print(f"[Producer] jump to TAIL {curr:0{n}b}")
            else:
                curr = random.choice(list(remaining))
                if verbose:
                    print(f"[Producer] jump to ANY unvisited {curr:0{n}b}")

            # Continue — do NOT emit curr here

        # Case 2: normal new state
        if verbose:
            print(f"[Producer] yield {curr:0{n}b}")

        yield int_to_vec(curr)

        # Mark visited
        visited.add(curr)
        prev = curr

        # Nonlinear update
        curr = nonlinear_step(curr)


# ================================================================
# ===================  Nonlinear Recoverer  ======================
# ================================================================


class NonlinearRecoverer(Recoverer):
    """Recover A from transitions (x → y = A F(x))."""

    def __init__(self, n, m, verbose=False):
        self.n = n
        self.m = m
        self.verbose = verbose

        self.prev = None
        self.transitions = set()
        self.recovered_index = None

        self.masks = generate_monomial_masks(n, m)

    # --------------------------------------
    def feed(self, data):
        if data is None:
            if self.verbose:
                print("[Recoverer] GAP → reset prev")
            self.prev = None
            return self.recovered_index

        data = np.array(data, dtype=np.uint8) % 2

        # Zero separator
        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer] SEPARATOR (zero vector) → reset prev")
            self.prev = None
            return self.recovered_index

        # If we have a previous nonzero, record transition
        if self.prev is not None and not np.all(self.prev == 0):
            pair = (tuple(self.prev.tolist()), tuple(data.tolist()))
            if pair not in self.transitions:
                if self.verbose:
                    print(
                        f"[Recoverer] ADD transition: {self.prev.tolist()} → {data.tolist()}"
                    )
                self.transitions.add(pair)
            else:
                if self.verbose:
                    print("[Recoverer] Transition duplicate, ignoring")
        self.prev = data.copy()

        # If already recovered
        if self.recovered_index is not None:
            return self.recovered_index

        # Need at least m transitions to hope for M×M matrix
        if len(self.transitions) < self.m:
            if self.verbose:
                print(
                    f"[Recoverer] Not enough transitions yet ({len(self.transitions)}/{self.m})"
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
                f"\n[Recoverer] BEGIN collecting independent F(x); transitions={len(self.transitions)}"
            )

        for x_bits, y_bits in self.transitions:
            x = np.array(x_bits, dtype=np.uint8)
            Fx = anf_feature_vector(x, self.masks)
            y = np.array(y_bits, dtype=np.uint8)

            # Print the Fx for visibility
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
                    f"[Recoverer] Only found {count}/{self.m} independent F(x) — cannot solve yet."
                )
            return None

        # ======================================================
        # Build X (m×m) and Y (n×m)
        # ======================================================
        X = np.column_stack(UX)
        Y = np.column_stack(UY)

        if self.verbose:
            print(f"\n[Recoverer] BUILT MATRICES:")
            print(f"  X shape = {X.shape}")
            print(f"  Y shape = {Y.shape}")
            print(f"  X=\n{X}")
            print(f"  Y=\n{Y}")

        # ======================================================
        # Solve A = Y X^{-1}
        # ======================================================
        try:
            A = self._solve_mod2(Y, X)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("[Recoverer] X was singular → retry later")
            return None

        if self.verbose:
            print("[Recoverer] SUCCESS — solved A:")
            print(A)

        self.recovered_index = self._matrix_to_index(A)
        return self.recovered_index

    # --------------------------------------
    def _matrix_to_index(self, A):
        idx = 0
        for r in range(A.shape[0]):
            for c in range(A.shape[1]):
                idx = (idx << 1) | int(A[r, c])
        return idx

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

# theoretically can handle up to D = 2^(n*2^n), where M=2^n
# but in practice large M starts to fail
# TODO: explore why it fails
override_D = lambda n: 2 ** (n * (2 ** (n - 1)))


def _infer_m_from_d(n, d):
    """Given n and d = 2^(n*m), solve for m."""
    # d must be of form 2^(n*m)
    from math import log2

    power = log2(d)
    m = int(power / n)
    if 2 ** (n * m) != d:
        raise ValueError(f"d={d} is not 2^(n*m) for integer m")
    # must also satisfy m <= 2^n (max monomials)
    if m > (1 << n):
        raise ValueError("m > 2^n is impossible for ANF monomials")
    return m


def producer_constructor(index: int, n: int, d: int) -> Producer:
    """
    Automatically choose M such that d = 2^(n*M) and M ≤ 2^n.
    """
    m = _infer_m_from_d(n, d)
    return GeneratorProducer(nonlinear_producer(n, m, index, d))


def recoverer_constructor(n: int, d: int) -> Recoverer:
    """
    Automatically choose M such that d = 2^(n*M).
    """
    m = _infer_m_from_d(n, d)
    return NonlinearRecoverer(n, m)
