from abc import ABC, abstractmethod
import numpy as np

# ---------- Helpers ----------


def number_to_bits(num: int, n: int) -> np.ndarray:
    """Convert integer to binary array of length n (MSB first)."""
    return np.array([(num >> i) & 1 for i in reversed(range(n))], dtype=np.uint8)


def bits_to_number(bits: np.ndarray) -> int:
    """Convert binary array to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def robust_soliton_probs(B, c=0.1, delta=0.5):
    """Compute Robust Soliton distribution for fountain code degrees."""
    R = c * np.log(B / delta) * np.sqrt(B)
    probs = np.zeros(B)
    for k in range(1, B + 1):
        probs[k - 1] = 1.0 / k if k < B else 1.0 / B
    probs /= probs.sum()
    return probs


def sample_degree(rng, cdf):
    u = rng.random()
    for i, p in enumerate(cdf):
        if u <= p:
            return i + 1
    return len(cdf)


# ---------- Deterministic RNG for reproducibility ----------


class XorShift64:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        x = self.state
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        self.state = x
        return x & 0xFFFFFFFFFFFFFFFF

    def random(self):
        return (self.next() & 0xFFFFFFFF) / 0x100000000

    def choice(self, lst, k):
        lst = list(lst)
        out = []
        for _ in range(k):
            i = int(self.random() * len(lst))
            out.append(lst.pop(i))
        return out


# ---------- Producer / Recoverer ----------


class Producer(ABC):
    @abstractmethod
    def generate(self) -> np.ndarray:
        """returns ndarray of shape (N,)"""


class Recoverer(ABC):
    @abstractmethod
    def feed(self, data: np.ndarray | None) -> int | None:
        """return reconstructed number 0..D-1 or None"""


class FountainProducer(Producer):
    def __init__(self, index: int, n: int, d: int):
        self.n = n  # number of bits per number
        self.d = d  # number of packets to produce (not really used)
        self.index = index
        self.rng = XorShift64(index + 0x12345678ABCDEF01)
        self.cdf = np.cumsum(robust_soliton_probs(n))
        # The number we want to encode
        self.number = index

    def generate(self) -> np.ndarray:
        bits = number_to_bits(self.number, self.n)
        # XOR a random small subset of bits
        deg = sample_degree(self.rng, self.cdf)
        indices = self.rng.choice(range(self.n), deg)
        packet = np.zeros(self.n, dtype=np.uint8)
        for i in indices:
            packet[i] ^= bits[i]
        return packet


class FountainRecoverer(Recoverer):
    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d
        self.B = n
        self.basis = {}  # pivot -> (row_int, rhs)

    def _insert_row(self, row: np.ndarray):
        """Gaussian elimination over GF(2) using integers."""
        row_int = int("".join(str(b) for b in row), 2)
        rhs = 1
        for pivot in sorted(self.basis.keys(), reverse=True):
            if (row_int >> pivot) & 1:
                row_int ^= self.basis[pivot][0]
                rhs ^= self.basis[pivot][1]
        if row_int == 0:
            return
        pivot = row_int.bit_length() - 1
        self.basis[pivot] = (row_int, rhs)

    def feed(self, data: np.ndarray | None) -> int | None:
        if data is None:
            # Interrupt, ignore
            return None
        self._insert_row(data)
        if len(self.basis) < self.B:
            return None  # not enough info yet
        # Back-substitute to recover bits
        x_bits = [0] * self.B
        for pivot in sorted(self.basis.keys()):
            row_int, rhs = self.basis[pivot]
            s = rhs
            mask = row_int & ~(1 << pivot)
            for j in range(self.B):
                if (mask >> j) & 1:
                    s ^= x_bits[j]
            x_bits[pivot] = s
        # Convert bits to integer
        return bits_to_number(np.array(x_bits, dtype=np.uint8))


# ---------- Constructor functions ----------


def producer_constructor(index: int, n: int, d: int) -> Producer:
    return FountainProducer(index, n, d)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    return FountainRecoverer(n, d)
