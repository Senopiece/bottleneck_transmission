import numpy as np

## ======================================================================================
## int <-> bits
## ======================================================================================


def bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def int_to_bits(value: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.bool_)
    bits = np.empty(length, dtype=np.bool_)
    for i in range(length):
        shift = length - 1 - i
        bits[i] = (value >> shift) & 1
    return bits


## ======================================================================================
## uint <-> bool array
## ======================================================================================


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


## ======================================================================================
## Message encoding
## ======================================================================================


def make_message_vector(message: np.ndarray, m: int, q: int) -> np.ndarray:
    """
    Convert message bits to vector in GF(q).

    Args:
        message: bits of the message
        m: number of symbols
        q: alphabet size
    Returns:
        vector as np.ndarray of shape (m,), dtype=np.uint16
    """
    int_msg = bits_to_int(message)

    vec = np.empty(m, dtype=np.uint16)

    for i in range(m):
        d = int_msg % q  # GF(q) elements
        int_msg //= q
        vec[i] = np.uint16(d)

    return vec


def message_from_message_vector(
    message_vector: np.ndarray,
    message_bitsize: int,
    q: int,
):
    """
    Convert message vector in GF(q) back to message bits.

    Args:
        message_vector: np.ndarray of shape (m,), dtype=np.uint16
        message_bitsize: expected size of the out message
        q: alphabet size
    Returns:
        message bits as np.ndarray of shape (message_bitsize,), dtype=np._bool
    """
    int_msg = 0

    for i in range(message_vector.shape[0] - 1, -1, -1):
        int_msg = int_msg * q + int(message_vector[i])

    return int_to_bits(int_msg, message_bitsize)
