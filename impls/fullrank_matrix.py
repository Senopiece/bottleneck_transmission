import numpy as np
from linal import (
    unrank_fullrank_matrix,
    rank_fullrank_matrix,
    count_fullrank_matrices,
    add_to_basis_rref,
)
import random


# TODO: investigate if we use primitive matrices will it transmit data faster?
#     (i) determine how to int <-> primitive matrix
#     (ii) utilize the same generation idea
#     (iii) make a sophisticated recoverer that is utilizing information that it will recover a primitive matrix, hence is faster than general full-rank matrix recovery
#     (iv) investigate is there a way to make better generation
# TODO: akin to primitive matrices, investigate using other special classes of full-rank matrices that have nice properties (e.g. circulant matrices, symmetric matrices, unitary matrices, orthogonal matrices, upper/lower triangular matrices, permutation matrices, positive definite, positive semi-definite/definite, lowrank but consists of multiplication of smaller fullrank matrices (through svd), normal matrices, hankel matrices, toeplitz matrices, matrices with some entries fixed to 0/1, etc...)
# TODO: generalize to be able to transfer arbitary amount of data
#      - fat matrices to give more information
#      - xor of specific matrix combinations
#      - some known matrix entries to reduce the amount of information to transfer
#      - resulting vector is concat of results of multiple smaller matrices (e.g. 3x3 and 2x2) to reduce the amount of information to transfer
# TODO: investigate is there a way to make sequences that they provide linearly independent vectors faster? like maybe insert more zeros, but choose such a patches that there are more linearly independent vectors?
class Producer:
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

        # --- Precompute all orbits under x → A x (mod 2) ---
        self.orbits: list[list[np.ndarray]] = []
        self._find_orbits()

        # --- Initialize traversal state ---
        self.orbit_order = list(range(len(self.orbits)))
        self.current_orbit_idx = 0
        self.current_orbit_pos = 0
        self.current_offset = random.randrange(len(self.orbits[self.orbit_order[0]]))
        self.after_separator = False
        self.cycle_count = 0

        if self.verbose:
            print(f"[Producer] A=\n{self.A}")
            print(f"[Producer] Found {len(self.orbits)} orbits total.")
            for i, orb in enumerate(self.orbits):
                print(f"  Orbit {i}: length={len(orb)}")
            print(f"[Producer] Starting order: {self.orbit_order}")

    # ------------------------------------------------------------
    def _find_orbits(self):
        """Compute all distinct orbits under x → A x (mod 2)."""
        all_states = [
            np.array(
                [(i >> (self.n - 1 - b)) & 1 for b in range(self.n)], dtype=np.uint8
            )
            for i in range(1, 2**self.n)
        ]
        seen = set()
        for s in all_states:
            if tuple(s) in seen:
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
    def _shuffle_orbits(self):
        """Reshuffle orbit order for the next global cycle."""
        random.shuffle(self.orbit_order)
        self.cycle_count += 1
        if self.verbose:
            print(
                f"[Producer] 🔀 Shuffled orbit order for cycle #{self.cycle_count}: {self.orbit_order}"
            )

    # ------------------------------------------------------------
    def generate(self) -> np.ndarray:
        """Continuously output all orbits, shuffled each full pass."""
        if self.after_separator:
            # Move to next orbit in current order
            self.current_orbit_idx += 1
            if self.current_orbit_idx >= len(self.orbit_order):
                # End of pass — reshuffle orbit order
                self._shuffle_orbits()
                self.current_orbit_idx = 0

            # Random offset for the new orbit
            orbit_id = self.orbit_order[self.current_orbit_idx]
            orbit_len = len(self.orbits[orbit_id])
            self.current_offset = random.randrange(orbit_len)
            self.current_orbit_pos = 0
            self.after_separator = False
            if self.verbose:
                print(
                    f"[Producer] → switched to orbit {orbit_id} (offset={self.current_offset})"
                )

        orbit_id = self.orbit_order[self.current_orbit_idx]
        orbit = self.orbits[orbit_id]
        orbit_len = len(orbit)

        # Output within current orbit
        if self.current_orbit_pos <= orbit_len:
            idx = (self.current_offset + self.current_orbit_pos) % orbit_len
            vec = orbit[idx]
            self.current_orbit_pos += 1
            if self.verbose:
                print(f"[Producer] out={''.join(map(str, vec))} (orbit {orbit_id})")
            return vec.copy()

        # After orbit end → emit zero separator
        else:
            self.after_separator = True
            if self.verbose:
                print(f"[Producer] → orbit separator (00000)")
            return np.zeros(self.n, dtype=np.uint8)


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
