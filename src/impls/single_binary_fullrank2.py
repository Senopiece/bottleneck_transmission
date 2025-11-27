import numpy as np
import random
from itertools import product
from math import prod

from ._interface import Producer, Recoverer

# ---------- Helper functions ----------


def vector_to_int(bits):
    """Convert a sequence of binary digits into an integer bitmask.

    The first element of ``bits`` becomes the most-significant bit in the
    returned integer. This is a small utility used throughout the module to
    represent binary column vectors compactly as Python ints.

    Args:
        bits: Iterable of 0/1 integers. Example: [1, 0, 1, 1].

    Returns:
        An int whose binary representation matches the input sequence.
        Example: [1, 0, 1, 1] -> 0b1011 (decimal 11).
    """
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def all_vectors(n):
    """Return all non-zero binary vectors of length ``n`` as integer masks.

    Each vector is encoded as an int using the same convention as
    :func:`vector_to_int` (first element -> most-significant bit). The
    zero vector is excluded because many algorithms here operate on
    independent/non-zero choices.

    Args:
        n: Length of the binary vectors.

    Returns:
        A list of ints representing all 2**n - 1 non-zero binary vectors.
    """
    return [vector_to_int(bits) for bits in product([0, 1], repeat=n) if any(bits)]


def add_to_basis_rref(v, basis):
    """Insert a vector into a basis kept in row-reduced echelon style.

    The basis is represented as a list of integers. Each integer has its
    pivot at its highest set bit. The function performs the reduction of
    ``v`` by the current basis, tests independence, and if independent,
    updates the basis so it remains in reduced echelon form.

    Args:
        v: Integer encoding of the vector to add.
        basis: List[int] representing the current RREF basis. Rows are
            expected to be sorted in descending pivot order (largest pivot
            first) but the function will maintain this ordering for the
            returned basis.

    Returns:
        A tuple (is_independent, new_basis):
        - is_independent: bool, True if ``v`` was independent of ``basis``.
        - new_basis: list[int], the updated basis (unchanged if dependent).

    Notes:
        The routine works purely with bitwise operations. It behaves like
        Gaussian elimination over GF(2) where each int is a row/column
        vector packed into bits.
    """
    # Reduce v by existing basis rows (eliminate pivots)
    for r in basis:
        pivot = 1 << (r.bit_length() - 1)
        if v & pivot:
            v ^= r

    # If v reduced to zero, it was dependent
    if v == 0:
        return False, basis

    # v is independent: eliminate its pivot from existing rows and append
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
    """Check whether vector ``v`` lies in the linear span of ``basis``.

    Args:
        v: Integer-encoded vector to test.
        basis: List[int] representing an RREF basis.

    Returns:
        True if ``v`` can be written as an XOR-combination of rows in
        ``basis`` (i.e., v is in span(basis)), False otherwise.
    """
    for r in basis:
        pivot = 1 << (r.bit_length() - 1)
        if v & pivot:
            v ^= r
    return v == 0


# ---------- rank / unrank utilities ----------


def unrank_fullrank_matrix(n, index):
    """Construct the ``index``-th full-rank binary NxN matrix (by columns).

    The algorithm enumerates all sequences of n non-zero column vectors
    that together form a full-rank matrix. At each column step it considers
    the set of vectors that are not in the span of previously chosen
    columns and selects one by taking ``index`` in a mixed-radix system.

    Args:
        n: Matrix size (number of rows and columns).
        index: Non-negative integer index in the range
            [0, count_fullrank_matrices(n)).

    Returns:
        cols: List[int] of length n where each int encodes a column vector.

    Notes:
        The output is a list of integer-encoded columns (not a nested list
        of bits). Use :func:`rank_fullrank_matrix` to invert this mapping.
    """
    allv = all_vectors(n)
    basis = []
    cols = []
    for step in range(n):
        # All choices that are still independent from the current basis
        valid = [v for v in allv if not in_span(v, basis)]
        choice = valid[index % len(valid)]
        index //= len(valid)
        cols.append(choice)
        _, basis = add_to_basis_rref(choice, basis)
    return cols


def rank_fullrank_matrix(n, cols):
    """Compute the unique index of a full-rank binary NxN matrix.

    This is the inverse of :func:`unrank_fullrank_matrix`. For an input
    list of integer-encoded columns this function computes the mixed-radix
    digits that identify the matrix in the enumeration and returns the
    reconstructed integer index.

    Args:
        n: Matrix size.
        cols: List[int] of length n containing integer-encoded columns that
            together form a full-rank matrix.

    Returns:
        Integer index corresponding to the given sequence of columns.
    """
    allv = all_vectors(n)
    basis = []
    digits = []
    bases = []

    for v in cols:
        valid = [x for x in allv if not in_span(x, basis)]
        pos = valid.index(v)
        digits.append(pos)
        bases.append(len(valid))
        _, basis = add_to_basis_rref(v, basis)

    # Fold the mixed-radix digits into a single integer index (right-to-left)
    i = digits[-1]
    for k in range(n - 2, -1, -1):
        i = digits[k] + bases[k] * i
    return i


def count_fullrank_matrices(n):
    """Return the number of full-rank binary NxN matrices.

    The count equals product_{i=0..n-1} (2^n - 2^i). This counts the number of
    ordered column lists that form a basis (i.e., full-rank matrices).

    Args:
        n: Matrix dimension.

    Returns:
        Integer count of full-rank NxN binary matrices.
    """
    return prod(2**n - 2**i for i in range(n))


class FullrankProducer(Producer):
    """
    Producer that:

    while True:
        for orbit in (orbits - zero orbit):
            x = rand(orbit)  # choose random starting vector from the orbit
            emitted = []
            while x not in columnspace(emitted):
                yield x
                emitted.append(x)
                x = A x
            yield x      # first dependent vector
            yield 0      # separator
    """

    def __init__(self, n: int, index: int, verbose: bool = False):
        self.n = n
        self.index = index
        self.verbose = verbose

        # --- Build generator matrix A from index ---
        self.cols = unrank_fullrank_matrix(n, index)
        self.A = np.array(
            [[(col >> (n - 1 - r)) & 1 for col in self.cols] for r in range(n)],
            dtype=np.uint8,
        )

        # --- Precompute all non-zero orbits under x -> A x (mod 2) ---
        self.orbits: list[list[np.ndarray]] = []
        self._find_orbits()  # only non-zero states are used

        # We will iterate over these orbit indices in a fixed cycle
        self.orbit_ids = list(range(len(self.orbits)))

        # --- State for current orbit run ---
        self.current_orbit_idx: int = 0  # index in orbit_ids
        self.current_orbit_pos: int = 0  # step along orbit from start offset
        self.start_offset: int = 0  # random start index in current orbit
        self.basis_masks: list[int] = (
            []
        )  # RREF basis for "emitted" vectors (as int masks)
        self.separator_pending: bool = (
            False  # if True, next call emits zero + moves to next orbit
        )

        # Initialize first orbit run
        if self.orbit_ids:
            self._start_new_orbit_run()

        if self.verbose:
            print(f"[Producer] A=\n{self.A}")
            print(f"[Producer] Found {len(self.orbits)} non-zero orbits total.")
            for i, orb in enumerate(self.orbits):
                print(f"  Orbit {i}: length={len(orb)}")
            print(f"[Producer] Initial orbit order: {self.orbit_ids}")

    # ------------------------------------------------------------
    def _find_orbits(self):
        """Compute all distinct non-zero orbits under x -> A x (mod 2)."""
        all_states = [
            np.array(
                [(i >> (self.n - 1 - b)) & 1 for b in range(self.n)],
                dtype=np.uint8,
            )
            for i in range(1, 2**self.n)  # skip zero state
        ]
        seen = set()
        for s in all_states:
            key = tuple(s)
            if key in seen:
                continue

            orbit = []
            x = s.copy()
            while True:
                orbit.append(x.copy())
                seen.add(tuple(x))
                x_next = (self.A @ x) % 2
                if np.array_equal(x_next, orbit[0]):
                    break
                if tuple(x_next) in seen:
                    break
                x = x_next

            self.orbits.append(orbit)

    # ------------------------------------------------------------
    def _start_new_orbit_run(self):
        """Prepare state for a new 'emitted' run on the current orbit."""
        orbit_id = self.orbit_ids[self.current_orbit_idx]
        orbit_len = len(self.orbits[orbit_id])
        self.start_offset = random.randrange(orbit_len)
        self.current_orbit_pos = 0
        self.basis_masks = []
        self.separator_pending = False

        if self.verbose:
            print(
                f"[Producer] -> starting new run on orbit {orbit_id}, "
                f"start_offset={self.start_offset}"
            )

    # ------------------------------------------------------------
    def _vec_to_mask(self, vec: np.ndarray) -> int:
        """Pack a binary vector (MSB = vec[0]) into an int."""
        m = 0
        for b in vec:
            m = (m << 1) | int(b)
        return m

    # ------------------------------------------------------------
    def generate(self) -> np.ndarray:
        """
        Streaming implementation of:

            x = rand(orbit)
            emitted = []
            while x not in colspace(emitted):
                yield x
                emitted.append(x)
                x = A x
            yield x      # first dependent vector
            yield 0      # separator
        """
        # If a separator is pending, emit zero and move to the next orbit
        if self.separator_pending:
            self.separator_pending = False

            # Advance to next orbit in the fixed order
            self.current_orbit_idx += 1
            if self.current_orbit_idx >= len(self.orbit_ids):
                self.current_orbit_idx = 0  # wrap around

            if self.orbit_ids:
                self._start_new_orbit_run()

            if self.verbose:
                print(f"[Producer] out={'0' * self.n} (separator)")

            return np.zeros(self.n, dtype=np.uint8)

        # Normal step: emit a vector from the current orbit
        if not self.orbit_ids:
            # Degenerate case (no non-zero orbits) – just emit zeros forever
            return np.zeros(self.n, dtype=np.uint8)

        orbit_id = self.orbit_ids[self.current_orbit_idx]
        orbit = self.orbits[orbit_id]
        orbit_len = len(orbit)

        idx = (self.start_offset + self.current_orbit_pos) % orbit_len
        x = orbit[idx].copy()

        # Check if x is in the span of 'emitted' via RREF basis
        mask = self._vec_to_mask(x)
        if in_span(mask, self.basis_masks):
            # First dependent vector:
            #   yield x now, then on next call yield zero and switch orbit
            self.separator_pending = True
            if self.verbose:
                print(
                    f"[Producer] out={''.join(map(str, x))} "
                    f"(orbit {orbit_id}, dependent -> separator next)"
                )
            return x

        # Independent: add to basis, advance along orbit, emit x
        indep, new_basis = add_to_basis_rref(mask, self.basis_masks)
        # indep should always be True here; if not, something is inconsistent
        if not indep and self.verbose:
            print("[Producer] WARNING: expected independent vector but got dependent.")

        self.basis_masks = new_basis
        self.current_orbit_pos += 1

        if self.verbose:
            print(
                f"[Producer] out={''.join(map(str, x))} "
                f"(orbit {orbit_id}, independent)"
            )

        return x


class FullrankRecoverer(Recoverer):
    def __init__(self, n: int, verbose: bool = False):
        self.n = n
        self.verbose = verbose
        self.prev = None
        # Store all unique transitions as tuples of tuples
        self.transitions: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        self.recovered_index = None

    def feed(self, data: np.ndarray | None) -> int | None:
        """Feed a new packet or None (gap). Returns recovered index or None."""
        if data is None:
            if self.verbose:
                print("[Recoverer] GAP -> reset prev")
            self.prev = None
            return self.recovered_index

        data = np.array(data, dtype=np.uint8) % 2

        # Handle explicit reset marker (zero packet)
        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer] Zero packet -> segment break")
            self.prev = None
            return self.recovered_index

        # Record a transition (prev -> data)
        if self.prev is not None and not np.all(self.prev == 0):
            key = (tuple(self.prev.tolist()), tuple(data.tolist()))
            if key not in self.transitions:
                self.transitions.add(key)
                if self.verbose:
                    print(
                        f"[Recoverer] + Added transition #{len(self.transitions)}: "
                        f"{''.join(map(str, self.prev))} -> {''.join(map(str, data))}"
                    )
            elif self.verbose:
                print(
                    f"[Recoverer]   (duplicate transition ignored, total={len(self.transitions)})"
                )

        self.prev = data.copy()

        # Skip solving if already recovered
        if self.recovered_index is not None:
            return self.recovered_index

        # Try recovery once we have enough distinct transitions
        if len(self.transitions) < self.n:
            if self.verbose:
                print(
                    f"[Recoverer] Transitions collected: {len(self.transitions)}/{self.n}"
                )
            return None

        # Collect up to N linearly independent transitions
        us, ws = [], []
        basis = []  # list of int-encoded columns (GF(2) RREF)
        for u_bits, w_bits in self.transitions:
            u = np.array(u_bits, dtype=np.uint8)
            # treat column as int mask for basis mgmt
            umask = 0
            for b in u:
                umask = (umask << 1) | int(b)
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

        # Solve for A
        U = np.column_stack(us) % 2
        W = np.column_stack(ws) % 2

        try:
            A = self._solve_mod2(W, U)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("[Recoverer] Singular transition matrix, retrying later")
            return None

        cols = [int("".join(str(b) for b in A[:, j]), 2) for j in range(self.n)]
        idx = rank_fullrank_matrix(self.n, cols)
        self.recovered_index = idx
        if self.verbose:
            print(f"[Recoverer] Recovered index={idx}")
        return idx

    @staticmethod
    def _solve_mod2(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Solve Y = A X for A over GF(2)."""
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


override_D = lambda n: count_fullrank_matrices(n)


def producer_constructor(index: int, n: int, d: int) -> Producer:
    assert d == count_fullrank_matrices(n)
    return FullrankProducer(n, index)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    assert d == count_fullrank_matrices(n)
    return FullrankRecoverer(n)
