import numpy as np
import random

from ._interface import Producer, Recoverer

# ---------- Helper: bit-vector utilities ----------


def vector_to_int(bits) -> int:
    """Convert a sequence of bits to an int (MSB first)."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


# ---------- Polynomial arithmetic over GF(2) ----------


def poly_deg(p: int) -> int:
    """Degree of polynomial p(x) over GF(2) encoded as bits."""
    return p.bit_length() - 1


def poly_mul(a: int, b: int) -> int:
    """Multiply two polynomials over GF(2) (bit-encoded)."""
    res = 0
    while b:
        if b & 1:
            res ^= a
        a <<= 1
        b >>= 1
    return res


def poly_mod(f: int, g: int) -> int:
    """Return f mod g over GF(2), both bit-encoded."""
    dg = poly_deg(g)
    while f and (f.bit_length() - 1) >= dg:
        shift = (f.bit_length() - 1) - dg
        f ^= g << shift
    return f


def poly_gcd(a: int, b: int) -> int:
    """GCD of polynomials a, b over GF(2)."""
    while b:
        a, b = b, poly_mod(a, b)
    return a


def poly_powmod(base: int, exp: int, mod: int) -> int:
    """Compute base**exp mod mod over GF(2)."""
    res = 1
    while exp > 0:
        if exp & 1:
            res = poly_mod(poly_mul(res, base), mod)
        base = poly_mod(poly_mul(base, base), mod)
        exp >>= 1
    return res


def factor_int(m: int) -> dict[int, int]:
    """Prime factorization of integer m (small, for 2^n-1)."""
    factors: dict[int, int] = {}
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors[d] = factors.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
    return factors


# ---------- Primitive polynomial & primitive matrix generation ----------


def is_irreducible(poly: int, n: int) -> bool:
    """
    Check irreducibility of a monic polynomial of degree n over GF(2).

    Uses the standard gcd(x^{2^k} - x, poly) test.
    """
    # Must be degree n, monic, and have non-zero constant term
    if poly_deg(poly) != n or not (poly & 1) or not (poly & (1 << n)):
        return False

    x = 0b10  # polynomial 'x'
    # For k = 1..n-1: gcd(x^{2^k} - x, poly) must be 1
    for k in range(1, n):
        h = poly_powmod(x, 1 << k, poly)  # x^{2^k} mod poly
        g = poly_gcd(h ^ x, poly)
        if g != 1:
            return False

    # And x^{2^n} ≡ x (mod poly)
    h = poly_powmod(x, 1 << n, poly)
    return poly_mod(h ^ x, poly) == 0


def is_primitive(poly: int, n: int) -> bool:
    """
    Check if an irreducible degree-n polynomial over GF(2) is primitive.

    Primitive ⇔ order of x mod poly is 2^n - 1.
    """
    if not is_irreducible(poly, n):
        return False

    m = (1 << n) - 1
    factors = factor_int(m)
    x = 0b10  # polynomial 'x'

    # For each prime q | (2^n - 1), x^{(2^n - 1)/q} != 1 (mod poly)
    for q in factors:
        e = m // q
        h = poly_powmod(x, e, poly)
        if h == 1:
            return False

    # And x^{2^n - 1} == 1 (mod poly)
    h = poly_powmod(x, m, poly)
    if h != 1:
        return False

    return True


def primitive_companion_matrix(poly: int, n: int) -> np.ndarray:
    """
    Build the companion matrix of a primitive polynomial of degree n.

    poly(x) = x^n + a_{n-1} x^{n-1} + ... + a_0, encoded in bits.
    Companion matrix over GF(2):

        [0  0  ...  0  a_0]
        [1  0  ...  0  a_1]
        [0  1  ...  0  a_2]
        ...
        [0  0  ...  1  a_{n-1}]
    """
    assert poly_deg(poly) == n
    A = np.zeros((n, n), dtype=np.uint8)

    # a_0..a_{n-1}
    coeffs = [(poly >> i) & 1 for i in range(n)]

    for r in range(n):
        for c in range(n):
            if r == 0 and c == n - 1:
                A[r, c] = coeffs[0]
            elif r > 0 and c == r - 1:
                A[r, c] = 1
            elif r > 0 and c == n - 1:
                A[r, c] = coeffs[r]

    return A


def all_primitive_matrices(n: int) -> list[np.ndarray]:
    """
    Enumerate primitive n×n matrices as companion matrices of primitive polynomials.

    For typical usage (n=5) this is tiny (e.g. 6 matrices).
    """
    mats: list[np.ndarray] = []
    for p in range(1 << (n + 1)):  # all polynomials up to degree n
        if is_primitive(p, n):
            mats.append(primitive_companion_matrix(p, n))
    return mats


# Cache of all primitive matrices per n
_PRIMITIVE_CACHE: dict[int, list[np.ndarray]] = {}


def primitive_family(n: int, d: int) -> list[np.ndarray]:
    """
    Return the first d primitive n×n matrices (d ≤ total count).

    This is the 'alphabet' of allowed matrices. Indexing is 0..d-1.
    """
    if n not in _PRIMITIVE_CACHE:
        _PRIMITIVE_CACHE[n] = all_primitive_matrices(n)
    mats = _PRIMITIVE_CACHE[n]
    assert d <= len(
        mats
    ), f"Requested d={d}, but only {len(mats)} primitive matrices exist for n={n}"
    return mats[:d]


def num_primitive_matrices(n: int) -> int:
    """Total number of primitive n×n matrices in our family."""
    if n not in _PRIMITIVE_CACHE:
        _PRIMITIVE_CACHE[n] = all_primitive_matrices(n)
    return len(_PRIMITIVE_CACHE[n])


# ---------- Primitive Producer (single-orbit matrices only) ----------


class PrimitiveProducer(Producer):
    """
    Producer restricted to primitive matrices over GF(2).

    A is chosen as the index-th matrix from a fixed family of primitive matrices
    (companion matrices of primitive polynomials). For such A:

        - A is invertible over GF(2)
        - The map x -> A x has exactly one orbit of length (2^n - 1)
          on nonzero vectors, and 0 is fixed.

    Stream generation logic:

        visited = {0}
        curr = 0

        loop:
            if curr in visited:
                # Start of new orbit segment
                if not after_separator:
                    emit 0...0 (separator)
                else:
                    if all states visited:
                        visited = {0}
                    pick random nonzero unvisited state as new start
                    emit it, then move with A

            else:
                emit curr, step curr = A curr

    Zero packet is always a separator marker.
    """

    def __init__(self, n: int, index: int, d: int, verbose: bool = False):
        self.n = n
        self.verbose = verbose

        # Fix primitive family (first d primitive matrices)
        self.mats = primitive_family(n, d)
        assert 0 <= index < d, f"index={index} out of range for d={d}"
        self.index = index
        self.A = self.mats[index]

        # State for stream
        self.curr: int = 0
        self.visited: set[int] = {0}
        self.all_states: set[int] = set(range(1 << n))
        self.unvisited: set[int] = self.all_states - self.visited
        self.after_separator: bool = False

        if self.verbose:
            print(f"[Producer] n={n}, index={index}, d={d}")
            print("[Producer] A =\n", self.A)
            print(f"[Producer] primitive family size={len(self.mats)}")

    # ------------------------------------------------------------
    def _int_to_vec(self, x: int) -> np.ndarray:
        return np.array(
            [(x >> (self.n - 1 - b)) & 1 for b in range(self.n)], dtype=np.uint8
        )

    def _vec_to_int(self, v: np.ndarray) -> int:
        return vector_to_int(v)

    # ------------------------------------------------------------
    def generate(self) -> np.ndarray:
        """Emit an infinite stream of n-bit packets (np.uint8), 0-vector as separator."""
        # Need to start a new orbit segment
        if self.curr in self.visited:
            # First emit separator
            if not self.after_separator:
                self.after_separator = True
                if self.verbose:
                    print("[Producer] -> separator (0...0)")
                return np.zeros(self.n, dtype=np.uint8)

            # If we've covered all states, reset visitation (keep 0 visited)
            if not self.unvisited:
                if self.verbose:
                    print("[Producer] RESET visited: all states seen, new cycle")
                self.visited = {0}
                self.unvisited = self.all_states - self.visited

            # Pick a fresh non-zero start state from unvisited
            start = random.choice(list(self.unvisited))
            self.visited.add(start)
            self.unvisited.remove(start)

            vec = self._int_to_vec(start)
            next_vec = (self.A @ vec) % 2
            self.curr = self._vec_to_int(next_vec)
            self.after_separator = False

            if self.verbose:
                print(f"[Producer] start orbit at {start:0{self.n}b}")
            return vec.copy()

        # Normal continuation inside the orbit
        self.visited.add(self.curr)
        if self.curr in self.unvisited:
            self.unvisited.remove(self.curr)

        vec = self._int_to_vec(self.curr)
        next_vec = (self.A @ vec) % 2
        self.curr = self._vec_to_int(next_vec)
        self.after_separator = False

        if self.verbose:
            print(f"[Producer] cont orbit: {''.join(map(str, vec))}")
        return vec.copy()


# ---------- Primitive-aware Recoverer ----------


class PrimitiveRecoverer(Recoverer):
    """
    Recoverer assuming A is a primitive matrix chosen from a known finite family.

    - Family: first d primitive matrices of size n×n over GF(2)
    - Index: which matrix in this family was used

    Recovery strategy:

        - Maintain candidate indices for A in primitive family.
        - Each observed transition u -> w (non-zero, not crossing a separator)
          must satisfy w = A u for the true A.
        - For each new distinct transition, filter candidates by w == A u.
        - Once exactly one candidate remains, its index is the payload.

    This uses the primitive prior to collapse uncertainty very fast
    (for n=5, typically 1–3 transitions).
    """

    def __init__(self, n: int, d: int, verbose: bool = False):
        self.n = n
        self.verbose = verbose

        self.prev: np.ndarray | None = None
        self.transitions: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        self.recovered_index: int | None = None

        self.mats = primitive_family(n, d)
        # Candidates stored as indices into self.mats
        self.candidates: list[int] = list(range(len(self.mats)))

        if self.verbose:
            print(f"[Recoverer] n={n}, d={d}, candidates={len(self.candidates)}")

    # ------------------------------------------------------------
    def feed(self, data: np.ndarray | None) -> int | None:
        """
        Feed a new packet or None (gap).

        Returns:
            - recovered payload index (0..d-1) once identified
            - or None if not enough information yet.
        """
        # Gap (burst loss / channel erasure)
        if data is None:
            if self.verbose:
                print("[Recoverer] GAP -> reset prev")
            self.prev = None
            return self.recovered_index

        data = np.array(data, dtype=np.uint8) % 2

        # Zero packet is a separator (end of an orbit segment)
        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer] Zero packet -> segment break")
            self.prev = None
            return self.recovered_index

        # Process transition prev -> data if prev is valid non-zero
        if self.prev is not None and not np.all(self.prev == 0):
            u = self.prev
            w = data
            key = (tuple(u.tolist()), tuple(w.tolist()))
            if key not in self.transitions:
                self.transitions.add(key)

                # Filter primitive candidates using this constraint
                new_candidates: list[int] = []
                for idx in self.candidates:
                    A = self.mats[idx]
                    w_expected = (A @ u) % 2
                    if np.array_equal(w_expected, w):
                        new_candidates.append(idx)

                self.candidates = new_candidates

                if self.verbose:
                    print(
                        f"[Recoverer] Added transition {''.join(map(str, u))} -> "
                        f"{''.join(map(str, w))}, candidates={len(self.candidates)}"
                    )

                if len(self.candidates) == 1 and self.recovered_index is None:
                    self.recovered_index = self.candidates[0]
                    if self.verbose:
                        print(f"[Recoverer] Recovered index={self.recovered_index}")

        # Update prev
        self.prev = data.copy()
        return self.recovered_index


# ---------- Constructors for external benchmark harness ----------

override_D = lambda n: num_primitive_matrices(n)


def producer_constructor(index: int, n: int, d: int) -> Producer:
    """
    Map index ∈ [0, d) to a primitive n×n matrix and return its Producer.

    d must satisfy d ≤ total number of primitive matrices available
    (num_primitive_matrices(n)).
    """
    # sanity check (will also populate cache)
    assert d <= num_primitive_matrices(n)
    return PrimitiveProducer(n, index, d)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    """
    Recoverer for the same primitive family used by producer_constructor.

    n, d must match the parameters used to construct the producer.
    """
    assert d <= num_primitive_matrices(n)
    return PrimitiveRecoverer(n, d)
