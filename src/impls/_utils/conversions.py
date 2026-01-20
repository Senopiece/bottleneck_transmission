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
## Z Backbone vector work
## ======================================================================================


def make_zbackbone_vector(int_msg: int, m: int, n: int) -> np.ndarray:
    """
    Convert message index to backbone vector (vector with no zero components) in GF(2^n).

    Args:
        int_msg: integer in [0, q^m - 1], where q = 2^n - 1
        m: number of semisymbols
        n: packet bitsize
    Returns:
        Backbone vector as np.ndarray of shape (m,), dtype=np.uint16
    """
    q = (1 << n) - 1
    vec = np.empty(m, dtype=np.uint16)

    for i in range(m):
        d = (int_msg % q) + 1  # GF(2^n) elements indexed from 1
        int_msg //= q
        vec[i] = np.uint16(d)

    return vec


def message_from_zbackbone_vector(backbone: np.ndarray, n: int) -> int:
    """
    Convert backbone vector (vector with no zero components) in GF(2^n) to message index.

    Args:
        backbone: np.ndarray of shape (m,), dtype=np.uint16
        n: packet bitsize
    Returns:
        Integer in [0, q^m - 1], where q = 2^n - 1
    """
    q = (1 << n) - 1
    int_msg = 0

    for i in range(backbone.shape[0] - 1, -1, -1):
        int_msg = int_msg * q + (int(backbone[i]) - 1)

    return int_msg


## ======================================================================================
## Plain backbone vector work
## ======================================================================================


def make_backbone_vector(int_msg: int, m: int, n: int) -> np.ndarray:
    """
    Convert message index to backbone vector in GF(2^n - 1).

    Args:
        int_msg: integer in [0, q^m - 1], where q = 2^n - 1
        m: number of semisymbols
        n: packet bitsize
    Returns:
        Backbone vector as np.ndarray of shape (m,), dtype=np.uint16
    """
    q = (1 << n) - 1
    vec = np.empty(m, dtype=np.uint16)

    for i in range(m):
        d = int_msg % q  # GF(2^n - 1) elements
        int_msg //= q
        vec[i] = np.uint16(d)

    return vec


def message_from_backbone_vector(backbone: np.ndarray, n: int) -> int:
    """
    Convert backbone vector in GF(2^n - 1) to message index.

    Args:
        backbone: np.ndarray of shape (m,), dtype=np.uint16
        n: packet bitsize
    Returns:
        Integer in [0, q^m - 1], where q = 2^n - 1
    """
    q = (1 << n) - 1
    int_msg = 0

    for i in range(backbone.shape[0] - 1, -1, -1):
        int_msg = int_msg * q + int(backbone[i])

    return int_msg


## ======================================================================================
## Raw backbone vector work
## ======================================================================================


def make_rbackbone_vector(message: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Convert message bits to backbone vector in GF(2^n).

    Args:
        message: bits
        m: number of symbols
        n: symbol bitsize
    Returns:
        Backbone vector as np.ndarray of shape (m,), dtype=np.uint16
    """
    L = message.size
    assert L <= m * n

    backbone = np.zeros(m, dtype=np.uint16)

    bit_idx = 0
    for i in range(m):
        v = np.uint16(0)
        for j in range(n):
            v <<= 1
            if bit_idx < L and message[bit_idx]:
                v |= 1
            bit_idx += 1
        backbone[i] = v

    return backbone


def message_from_rbackbone_vector(
    backbone: np.ndarray,
    message_bitsize: int,
    m: int,
    n: int,
):
    message = np.empty(message_bitsize, dtype=np.bool_)
    bit_idx = 0
    for i in range(m):
        v = backbone[i]
        for j in range(n - 1, -1, -1):
            if bit_idx >= message_bitsize:
                break
            message[bit_idx] = ((v >> j) & 1) != 0
            bit_idx += 1
        if bit_idx >= message_bitsize:
            break

    return message
