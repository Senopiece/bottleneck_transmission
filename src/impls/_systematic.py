from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from ._interface import Estimator, Packet, Message, Protocol, Config, Sampler


# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1


def _bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def _vector_to_int(bits: np.ndarray) -> int:
    value = 0
    for i, bit in enumerate(bits):
        value |= (int(bit) & 1) << i
    return value


def _add_to_basis_rref_vector(v_bits: np.ndarray, basis: list[int]):
    v = _vector_to_int(v_bits)
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


def _invert_gf2(matrix: np.ndarray) -> np.ndarray | None:
    rows, cols = matrix.shape
    if rows != cols:
        return None
    if rows == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    aug = np.concatenate([matrix.copy(), np.eye(rows, dtype=np.uint8)], axis=1)
    for i in range(rows):
        pivot = None
        for r in range(i, rows):
            if aug[r, i]:
                pivot = r
                break
        if pivot is None:
            return None
        if pivot != i:
            aug[[i, pivot]] = aug[[pivot, i]]
        for r in range(rows):
            if r != i and aug[r, i]:
                aug[r] ^= aug[i]
    return aug[:, rows:]


_MASK64 = (1 << 64) - 1
_MIX64_CONST = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    v = (value + _MIX64_CONST) & _MASK64
    z = v
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    return z ^ (z >> 31)


def _hash_feature_vector(x_int: int, seed: int, rank: int) -> np.ndarray:
    if rank <= 0:
        return np.zeros(0, dtype=np.uint8)
    out = np.zeros(rank, dtype=np.uint8)
    state = (seed ^ (x_int * _MIX64_CONST)) & _MASK64
    idx = 0
    while idx < rank:
        state = _splitmix64(state)
        chunk = state
        take = min(64, rank - idx)
        for bit in range(take):
            out[idx + bit] = (chunk >> bit) & 1
        idx += take
    return out


def _seed_stream(rank: int, input_bits: int):
    state = (rank << 32) ^ (input_bits << 16) ^ _MIX64_CONST
    while True:
        state = _splitmix64(state)
        yield state


def _build_hash_basis(input_bits: int, rank: int, seed: int) -> bool:
    if rank <= 0:
        return True
    basis: list[int] = []
    for x in range(1 << input_bits):
        fx = _hash_feature_vector(x, seed, rank)
        indep, basis = _add_to_basis_rref_vector(fx, basis)
        if indep and len(basis) == rank:
            return True
    return False


@lru_cache(maxsize=None)
def _find_hash_seed(rank: int, input_bits: int, max_tries: int = 4096) -> int:
    stream = _seed_stream(rank, input_bits)
    for _ in range(max_tries):
        seed = next(stream)
        if _build_hash_basis(input_bits, rank, seed):
            return seed
    raise ValueError("Could not find hash seed with full-rank feature matrix")


def _normalize_payload(payload: Message, message_bitsize: int) -> np.ndarray:
    bits = np.array(payload, dtype=np.uint8).reshape(-1) & 1
    if bits.size != message_bitsize:
        raise ValueError(
            f"Expected payload of {message_bitsize} bits, got {bits.size} bits"
        )
    return bits


def _payload_to_matrix(
    message_bitsize: np.ndarray, output_bits: int, rank: int, zero_pad: int
) -> np.ndarray:
    capacity = output_bits * rank
    bits = np.array(message_bitsize, dtype=np.uint8) & 1
    if bits.size > capacity:
        raise ValueError("payload exceeds available capacity")

    pad_len = capacity - bits.size
    if zero_pad != pad_len:
        raise ValueError("zero_pad mismatch")
    if pad_len:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])

    order = "F" if rank > output_bits else "C"
    return bits.reshape((output_bits, rank), order=order)


def _matrix_to_payload(
    matrix: np.ndarray, message_bitsize: int, rank: int, output_bits: int
) -> np.ndarray:
    if message_bitsize <= 0:
        return np.zeros(0, dtype=np.uint8)
    order = "F" if rank > output_bits else "C"
    flat = matrix.reshape(-1, order=order)
    return flat[:message_bitsize].copy()


def _ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return x.bit_length() - (1 if (x & (x - 1)) == 0 else 0)


def _select_layout(message_bitsize: int, packet_bitsize: int):
    """
    Selects a optimal layout given message_bitsize and packet_bitsize.

    Choose (rank, input_bits, output_bits, zero_pad) with:
      - output_bits = packet_bitsize - input_bits > 0
      - rank * output_bits >= message_bitsize
    Optimize lexicographically:
      1) rank (min)
      2) zero_pad (min)
      3) input_bits (min)

    Note: we would need to collect rank linearly inpependent observations to recover the original data. So we need it to be as small as possible.
    Also lowering the rank increases output_bits that increases partitioning so the reconstruction becomes more parellelizable.
    TODO: maybe sometimes it can be beneficial of having not the minimum possible rank, but achieving better results somehow (e.g. for 10,5 to use (5, 3, 2, 0) instead of (4, 2, 3, 0) - so theoretically its having lower rank, but the second option has better resilience from errors).
    TODO: maybe apply ml in search of optimal layouting
    """

    p = message_bitsize
    n = packet_bitsize

    best = None  # (rank, zero_pad, input_bits, output_bits)

    # A safe upper bound: if input_bits=0 allowed, rank=p is enough (when n>0).
    # We also cap rank by 2^n *something; but p is typically the meaningful bound.
    for rank in range(1, max(2, p + 1)):
        min_in = _ceil_log2(rank)

        # input_bits can range up to n-1 (must leave at least 1 output bit)
        for input_bits in range(min_in, n):
            output_bits = n - input_bits
            capacity = rank * output_bits
            if capacity < p:
                continue

            zero_pad = capacity - p
            cand = (rank, zero_pad, input_bits, output_bits)

            if best is None or cand[:3] < best[:3]:
                best = cand

        # rank is primary objective; once we've found any solution at this rank,
        # we can stop (because larger ranks are worse).
        if best is not None and best[0] == rank:
            break

    if best is None:
        return None

    rank, zero_pad, input_bits, output_bits = best
    return rank, input_bits, output_bits, zero_pad


class _SystematicSamplerState:
    def __init__(
        self,
        packet_bitsize: int,
        message_bitsize: int,
        payload: Message,
        layout: tuple[int, int, int, int],
        seed: int,
    ):
        self.n = packet_bitsize
        self.message_bitsize = message_bitsize
        self.rank, self.input_bits, self.output_bits, self.zero_pad = layout
        self.seed = seed

        message_bitsize_arr = _normalize_payload(payload, message_bitsize)
        self.A = _payload_to_matrix(
            message_bitsize_arr, self.output_bits, self.rank, self.zero_pad
        ).astype(np.uint8)

        payload_seed = _bits_to_int(message_bitsize_arr)
        seed_material = (
            (payload_seed << 24)
            ^ (packet_bitsize << 16)
            ^ (self.rank << 8)
            ^ (self.input_bits << 4)
            ^ 0x51D5A7
        )
        self.rng = random.Random(seed_material)

    def generate(self) -> Packet:
        if self.input_bits == 0:
            x = np.zeros(0, dtype=np.uint8)
        else:
            x = np.array(
                [self.rng.randrange(2) for _ in range(self.input_bits)],
                dtype=np.uint8,
            )

        fx = _hash_feature_vector(_vector_to_int(x), self.seed, self.rank)
        y = (self.A @ fx) % 2
        if self.input_bits == 0:
            packet = y
        else:
            packet = np.concatenate([x, y])
        return packet.astype(np.bool_, copy=False)


class _SystematicEstimatorState:
    def __init__(
        self,
        packet_bitsize: int,
        message_bitsize: int,
        layout: tuple[int, int, int, int],
        seed: int,
    ):
        self.n = packet_bitsize
        self.message_bitsize = message_bitsize
        self.rank, self.input_bits, self.output_bits, self.zero_pad = layout
        self.seed = seed

        self.basis: list[int] = []
        self.x_cols: list[np.ndarray] = []
        self.y_cols: list[np.ndarray] = []
        self.recovered_payload: np.ndarray | None = (
            np.zeros(0, dtype=np.bool_) if self.message_bitsize == 0 else None
        )

    def progress(self) -> float:
        if self.recovered_payload is not None:
            return 1.0
        if self.rank <= 0:
            return 0.0
        return min(1.0, len(self.x_cols) / self.rank)

    def feed(self, data: Packet | None) -> Message | None:
        if self.recovered_payload is not None:
            return self.recovered_payload
        if self.message_bitsize == 0:
            self.recovered_payload = np.zeros(0, dtype=np.bool_)
            return self.recovered_payload
        if data is None:
            return None

        packet = np.array(data, dtype=np.uint8).reshape(-1) & 1
        if packet.shape != (self.n,):
            raise ValueError("packet shape mismatch")

        x = packet[: self.input_bits]
        y = packet[self.input_bits :]
        fx = _hash_feature_vector(_vector_to_int(x), self.seed, self.rank)

        indep, new_basis = _add_to_basis_rref_vector(fx, self.basis)
        if not indep:
            return None
        self.basis = new_basis
        self.x_cols.append(fx)
        self.y_cols.append(y)

        if len(self.x_cols) < self.rank:
            return None

        X = np.column_stack(self.x_cols)
        Y = np.column_stack(self.y_cols)
        Xinv = _invert_gf2(X)
        if Xinv is None:
            return None

        A = (Y @ Xinv) % 2
        payload = _matrix_to_payload(
            A, self.message_bitsize, self.rank, self.output_bits
        )
        self.recovered_payload = payload.astype(np.bool_, copy=False)
        return self.recovered_payload


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = int(config.packet_bitsize)
    message_bitsize = int(config.message_bitsize)
    if message_bitsize < 0:
        raise ValueError("message_bitsize must be >= 0")

    max_bits = max_message_bitsize(packet_bitsize)
    if message_bitsize > max_bits:
        raise ValueError(
            f"Cannot encode payload with {message_bitsize} bits, max supported for "
            f"packet size {packet_bitsize} is {max_bits}"
        )

    layout = _select_layout(message_bitsize, packet_bitsize)
    if layout is None:
        raise ValueError(
            f"Cannot encode payload with {message_bitsize} bits and packet size {packet_bitsize}"
        )
    rank, input_bits, _, _ = layout
    seed = _find_hash_seed(rank, input_bits)

    def make_sampler(payload: Message) -> Sampler:
        sampler_state = _SystematicSamplerState(
            packet_bitsize, message_bitsize, payload, layout, seed
        )
        while True:
            yield sampler_state.generate()

    def make_estimator() -> Estimator:
        estimator_state = _SystematicEstimatorState(
            packet_bitsize, message_bitsize, layout, seed
        )
        packet = yield estimator_state.progress()
        while True:
            recovered = estimator_state.feed(packet)
            if recovered is not None:
                return recovered
            packet = yield estimator_state.progress()

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    """Return the maximum payload size (in bits) for packets of the given size."""
    return 2 ** (packet_bitsize - 1)
