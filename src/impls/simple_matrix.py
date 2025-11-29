import numpy as np

from ._interface import Producer, Recoverer, GeneratorProducer

# NOTE: fails to transmit some degenerate payloads e.g. D = 0


# ---------- Helper functions ----------


def vector_to_int(bits):
    """Convert a sequence of binary digits into an integer bitmask.

    The first element of ``bits`` becomes the most-significant bit in the
    returned integer. This is a small utility used throughout the module to
    represent binary column vectors compactly as Python ints.
    """
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def add_to_basis_rref(v, basis):
    """Insert a vector into a basis kept in row-reduced echelon style (GF(2)).

    basis: list[int], each int has its pivot at its highest set bit.
    Returns (is_independent, new_basis).
    """
    # Reduce v by existing basis rows
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


# ---------- new rank / unrank for *all* matrices ----------


def index_to_matrix(n: int, index: int) -> np.ndarray:
    """Map integer index in [0, 2^(n^2)) to an arbitrary binary n×n matrix.

    Bits are laid out in row-major order, MSB first.
    """
    max_val = 1 << (n * n)
    assert 0 <= index < max_val, "Index out of range for n×n matrix payload"

    bits = [(index >> k) & 1 for k in reversed(range(n * n))]
    A = np.array(bits, dtype=np.uint8).reshape((n, n))
    return A


def matrix_to_index(A: np.ndarray) -> int:
    """Inverse of index_to_matrix: encode an n×n binary matrix as an integer."""
    A = np.array(A, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix must be square"

    idx = 0
    for r in range(n):
        for c in range(n):
            idx = (idx << 1) | int(A[r, c] & 1)
    return idx


# ---------- Producer ----------


def fullrank_matrix_producer(n: int, index: int, d: int, verbose: bool = False):
    """
    Pure generator form of the pseudocode:

        visited = {0}
        curr = 0
        while True:
            if curr in visited:
                yield 0
                if all vectors visited:
                    visited = {0}
                curr = random unvisited
            yield curr
            visited.add(curr)
            curr = A @ curr % 2
    """

    assert d == 2 ** (n * n)

    import numpy as np
    import random
    from .simple_matrix import index_to_matrix

    A = index_to_matrix(n, index)

    # Helpers
    def int_to_vec(x: int) -> np.ndarray:
        return np.array([(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)

    def vec_to_int(v: np.ndarray) -> int:
        x = 0
        for b in v:
            x = (x << 1) | int(b)
        return x

    # State
    curr = 0
    visited = {0}
    all_states = set(range(2**n))
    unvisited = all_states - visited

    if verbose:
        print("[Producer] Using A =\n", A)

    # ========================
    #     MAIN LOOP
    # ========================
    while True:

        # ---------------------------------------------
        # curr already visited → separator + new start
        # ---------------------------------------------
        if curr in visited:
            # emit separator (always, as in pseudocode)
            if verbose:
                print(f"[Producer] separator at curr={curr}")
            yield np.zeros(n, dtype=np.uint8)

            # reset visit set if we exhausted everything
            if not unvisited:
                visited = {0}
                unvisited = all_states - visited
                if verbose:
                    print("[Producer] RESET visited set")

            # pick a fresh starting point
            curr = random.choice(tuple(unvisited))
            if verbose:
                print(f"[Producer] new start = {curr:0{n}b}")

        # ---------------------------------------------
        # emit current state
        # ---------------------------------------------
        v = int_to_vec(curr)
        yield v.copy()

        # mark visited
        visited.add(curr)
        unvisited.discard(curr)

        # ---------------------------------------------
        # next state: curr = A @ curr
        # ---------------------------------------------
        next_vec = (A @ v) % 2
        curr = vec_to_int(next_vec)

        if verbose:
            print(
                f"[Producer] {''.join(map(str, v))} -> {curr:0{n}b}, "
                f"visited={len(visited)}"
            )


# ---------- Recoverer ----------


class FullrankRecoverer(Recoverer):
    """
    Recoverer for arbitrary binary n×n matrices A.

    From the stream of n-bit packets (with zero separators and possible None gaps),
    we collect distinct transitions (u -> w) that correspond to u, w ≠ 0 and
    occur consecutively without a separator. These are guaranteed to satisfy w = A u.

    Once we have n linearly independent u's, we solve Y = A X over GF(2):

        X = [u_1, ..., u_n],  Y = [w_1, ..., w_n]
        A = Y X^{-1}  (in GF(2))

    Then we map A back to the payload integer by matrix_to_index(A).
    """

    def __init__(self, n: int, verbose: bool = False):
        self.n = n
        self.verbose = verbose

        self.prev: np.ndarray | None = None
        # Store all unique transitions as (u_bits, w_bits) tuples
        self.transitions: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        self.recovered_index: int | None = None

    # ------------------------------------------------------------
    def feed(self, data: np.ndarray | None) -> int | None:
        """Feed a new packet or None (gap). Returns recovered payload index or None."""
        # Handle burst loss / explicit gap
        if data is None:
            if self.verbose:
                print("[Recoverer] GAP -> reset prev")
            self.prev = None
            return self.recovered_index

        data = np.array(data, dtype=np.uint8) % 2

        # Zero packet -> separator between orbits
        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer] Zero packet -> segment break")
            self.prev = None
            return self.recovered_index

        # Record a transition (prev -> data) if we have a valid prev
        if self.prev is not None and not np.all(self.prev == 0):
            key = (tuple(self.prev.tolist()), tuple(data.tolist()))
            if key not in self.transitions:
                self.transitions.add(key)
                if self.verbose:
                    print(
                        f"[Recoverer] + transition #{len(self.transitions)}: "
                        f"{''.join(map(str, self.prev))} -> {''.join(map(str, data))}"
                    )
            elif self.verbose:
                print(
                    f"[Recoverer]   (duplicate transition ignored, total={len(self.transitions)})"
                )

        self.prev = data.copy()

        # If already recovered, just return the cached index
        if self.recovered_index is not None:
            return self.recovered_index

        # Need at least n distinct transitions to hope for full rank in the domain u's
        if len(self.transitions) < self.n:
            if self.verbose:
                print(
                    f"[Recoverer] Transitions collected: {len(self.transitions)}/{self.n}"
                )
            return None

        # Collect up to n linearly independent u's (columns) in GF(2)
        us: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        basis: list[int] = []  # RREF basis of chosen u's (as int bitmasks)

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

        # Solve Y = A X for A over GF(2)
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
        # Augment [X | I] and compute X^{-1} via Gauss–Jordan over GF(2)
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


# ---------- constructors for external benchmark harness ----------


override_D = lambda n: 2 ** (n * n)


def producer_constructor(index: int, n: int, d: int) -> Producer:
    # All n×n binary matrices are allowed: total payload size 2^(n^2)
    assert d == 2 ** (n * n)
    return GeneratorProducer(fullrank_matrix_producer(n, index, d))


def recoverer_constructor(n: int, d: int) -> Recoverer:
    assert d == 2 ** (n * n)
    return FullrankRecoverer(n)
