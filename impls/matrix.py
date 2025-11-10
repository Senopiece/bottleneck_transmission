import numpy as np
from linal import (
    unrank_fullrank_matrix,
    rank_fullrank_matrix,
    count_fullrank_matrices,
    add_to_basis_rref,
)
import random

# ============================================================
#  Producer that enumerates all orbits of A mod 2 in random order
# ============================================================


class Producer:
    def __init__(self, n: int, index: int, verbose: bool = False):
        self.n = n
        self.index = index
        self.verbose = verbose

        # --- Build generator matrix A from full-rank index ---
        self.cols = unrank_fullrank_matrix(n, index)
        self.A = np.array(
            [[(col >> (n - 1 - r)) & 1 for col in self.cols] for r in range(n)],
            dtype=np.uint8,
        )

        # --- Precompute all orbits (excluding zero vector) ---
        self.orbits: list[list[np.ndarray]] = []
        self._find_orbits()

        # --- Initialize state machine ---
        self.current_orbit_idx = 0
        self.current_orbit_pos = 0
        self.prev_orbit_idx = None
        self.sequence_exhausted = False

        if self.verbose:
            print(f"[Producer] A=\n{self.A}")
            print(f"[Producer] Total orbits found: {len(self.orbits)}")
            for i, orbit in enumerate(self.orbits):
                print(f"  Orbit {i}: length={len(orbit)}")

    # ------------------------------------------------------------
    def _find_orbits(self):
        """Compute all distinct orbits under x → A x (mod 2)."""
        all_states = [
            np.array(
                [(i >> (self.n - 1 - b)) & 1 for b in range(self.n)], dtype=np.uint8
            )
            for i in range(1, 2**self.n)
        ]  # exclude zero
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
                if np.array_equal(x_next, orbit[0]):  # closed orbit
                    break
                if tuple(x_next) in seen:  # safety guard
                    break
                x = x_next

            self.orbits.append(orbit)

    # ------------------------------------------------------------
    def _choose_next_orbit(self):
        """Randomly choose the next orbit, avoiding the previous one if possible."""
        num_orbits = len(self.orbits)
        if num_orbits == 1:
            return 0  # only one orbit exists
        candidates = [i for i in range(num_orbits) if i != self.prev_orbit_idx]
        return random.choice(candidates)

    # ------------------------------------------------------------
    def generate(self) -> np.ndarray | None:
        """Emit vectors orbit-by-orbit with 0 separators, switching randomly between orbits."""
        if self.sequence_exhausted:
            if self.verbose:
                print("[Producer] End of transmission.")
            return None

        # Case 1: emitting current orbit
        if self.current_orbit_pos < len(self.orbits[self.current_orbit_idx]):
            state = self.orbits[self.current_orbit_idx][self.current_orbit_pos]
            self.current_orbit_pos += 1
            if self.verbose:
                print(
                    f"[Producer] out={''.join(map(str, state))} (orbit {self.current_orbit_idx})"
                )
            return state.copy()

        # Case 2: finished current orbit → emit zero separator
        elif self.current_orbit_pos == len(self.orbits[self.current_orbit_idx]):
            self.current_orbit_pos += 1
            if self.verbose:
                print(f"[Producer] → orbit separator (00000)")
            return np.zeros(self.n, dtype=np.uint8)

        # Case 3: after separator, switch to next random orbit
        else:
            remaining = len(self.orbits)
            if remaining == 0:
                self.sequence_exhausted = True
                return None

            self.prev_orbit_idx = self.current_orbit_idx
            self.current_orbit_idx = self._choose_next_orbit()
            self.current_orbit_pos = 0

            if self.verbose:
                print(
                    f"[Producer] Switching orbit: {self.prev_orbit_idx} → {self.current_orbit_idx}"
                )

            # Recursively generate the first vector of the new orbit
            return self.generate()


# ============================================================
#  Recoverer with rank diagnostics
# ============================================================


class Recoverer:
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
                print("[Recoverer] GAP → reset prev")
            self.prev = None
            return self.recovered_index

        data = np.array(data, dtype=np.uint8) % 2

        # Handle explicit reset marker (zero packet)
        if np.all(data == 0):
            if self.verbose:
                print("[Recoverer] Zero packet → segment break")
            self.prev = None
            return self.recovered_index

        # Record a transition (prev → data)
        if self.prev is not None and not np.all(self.prev == 0):
            key = (tuple(self.prev.tolist()), tuple(data.tolist()))
            if key not in self.transitions:
                self.transitions.add(key)
                if self.verbose:
                    print(
                        f"[Recoverer] + Added transition #{len(self.transitions)}: "
                        f"{''.join(map(str, self.prev))} → {''.join(map(str, data))}"
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
                    f"[Recoverer] Rank {len(us)}/{self.n} → waiting for more independent transitions."
                )
            return None

        # Solve for A
        U = np.column_stack(us) % 2
        W = np.column_stack(ws) % 2

        try:
            A = self._solve_mod2(W, U)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("[Recoverer] ❌ Singular transition matrix, retrying later")
            return None

        cols = [int("".join(str(b) for b in A[:, j]), 2) for j in range(self.n)]
        idx = rank_fullrank_matrix(self.n, cols)
        self.recovered_index = idx
        if self.verbose:
            print(f"[Recoverer] ✅ Recovered index={idx}")
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


# ============================================================
#  Demo runner
# ============================================================

if __name__ == "__main__":
    n = 5
    total = count_fullrank_matrices(n)
    index = np.random.randint(0, total)
    print(f"[Producer] index={index}")

    prod = Producer(n, index, verbose=True)
    recv = Recoverer(n, verbose=True)

    for step in range(300):
        out = prod.generate()
        rec = recv.feed(out)
        if rec is not None:
            print(f"\n[✓] Matrix recovered after {step} packets! index={rec}")
            assert rec == index
            break
    else:
        print("\n⚠️ Did not recover within limit.")


def producer_constructor(index: int, n: int, d: int) -> Producer:
    assert d == count_fullrank_matrices(n)
    return Producer(n, index)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    assert d == count_fullrank_matrices(n)
    return Recoverer(n)
