import numpy as np


def bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def int_to_bits(value: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        shift = length - 1 - i
        bits[i] = (value >> shift) & 1
    return bits


def uint16_to_bool_array(x: np.uint16, n: int = 16) -> np.ndarray:
    """
    Convert a uint16 to an NDArray[np.bool_] of length n.
    bits[0] is the most significant bit.
    """
    out = np.empty(n, dtype=np.bool_)
    v = int(x)
    for i in range(n):
        shift = n - 1 - i
        out[i] = (v >> shift) & 1
    return out


def bool_array_to_uint16(bits: np.ndarray) -> np.uint16:
    """
    Convert NDArray[np.bool_] to uint16.
    bits[0] is the most significant bit.
    """
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint16(v)
