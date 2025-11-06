from itertools import product
from math import prod
from tqdm import tqdm

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


if __name__ == "__main__":
    n = 5
    for idx in tqdm(range(count_fullrank_matrices(n))):
        cols = unrank_fullrank_matrix(n, idx)
        assert rank_fullrank_matrix(n, cols) == idx, f"{idx} rank/unrank mismatch"
