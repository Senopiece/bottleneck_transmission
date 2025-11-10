import numpy as np
from linal import unrank_fullrank_matrix, rank_fullrank_matrix, count_fullrank_matrices


# ============================================================
#  Producer with cycle detection and verbose logging
# ============================================================


class Producer:
    def __init__(
        self, n: int, index: int, seed: np.ndarray | None = None, verbose: bool = False
    ):
        self.n = n
        self.index = index
        self.verbose = verbose

        self.cols = unrank_fullrank_matrix(n, index)
        self.A = np.array(
            [[(col >> (n - 1 - r)) & 1 for col in self.cols] for r in range(n)],
            dtype=np.uint8,
        )

        if seed is None:
            self.state = np.random.randint(0, 2, n, dtype=np.uint8)
        else:
            self.state = seed.copy()

        self.visited = {tuple(self.state)}
        self.reset_pending = False
        self.resets_done = 0

        if self.verbose:
            print(f"[Producer] A=\n{self.A}")
            print(f"[Producer] init state={''.join(map(str, self.state))}")

    def generate(self) -> np.ndarray:
        """Generate next vector, injecting a zero packet on cycle loopback."""
        if self.reset_pending:
            self.reset_pending = False
            if self.verbose:
                print(f"[Producer] → cycle reset marker (00000)")
            return np.zeros(self.n, dtype=np.uint8)

        next_state = (self.A @ self.state) % 2
        key = tuple(next_state)

        if key in self.visited:
            # cycle detected
            self.visited.clear()
            self.reset_pending = True
            self.resets_done += 1
            if self.verbose:
                print(
                    f"[Producer] Cycle detected after {len(self.visited)} states → inject 0, restart."
                )
            # choose a new random nonzero seed
            while True:
                self.state = np.random.randint(0, 2, self.n, dtype=np.uint8)
                if np.any(self.state):
                    break
            self.visited.add(tuple(self.state))
            return np.zeros(self.n, dtype=np.uint8)

        self.visited.add(key)
        self.state = next_state

        if self.verbose:
            print(f"[Producer] out={''.join(map(str, self.state))}")
        return self.state.copy()


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
        for u_bits, w_bits in self.transitions:
            u = np.array(u_bits, dtype=np.uint8)
            w = np.array(w_bits, dtype=np.uint8)
            # Add if increases rank
            if len(us) == 0:
                us.append(u)
                ws.append(w)
            else:
                rank_before = np.linalg.matrix_rank(np.column_stack(us) % 2)
                rank_after = np.linalg.matrix_rank(np.column_stack(us + [u]) % 2)
                if rank_after > rank_before:
                    us.append(u)
                    ws.append(w)
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
