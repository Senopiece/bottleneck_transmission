import numpy as np

from ._interface import Producer, Recoverer
from .shuffled_binary_matrix import (
    MatrixProducer,
    MatrixRecoverer,
    _init_global_defaults,
    add_to_basis_rref,
    index_to_matrix,
    matrix_to_index,
    vector_to_int,
)
from .single_binary_fullrank import (
    count_fullrank_matrices,
    rank_fullrank_matrix,
    unrank_fullrank_matrix,
)


# ---------- Helpers for full-rank matrix indexing ----------


def fullrank_index_to_matrix(n: int, index: int) -> np.ndarray:
    """Map payload index -> full-rank binary n x n matrix (columns MSB-first)."""
    cols = unrank_fullrank_matrix(n, index)
    return np.array(
        [[(col >> (n - 1 - r)) & 1 for col in cols] for r in range(n)],
        dtype=np.uint8,
    )


def _is_fullrank(A: np.ndarray) -> bool:
    """Check full-rank over GF(2) via RREF-style basis building."""
    n = A.shape[0]
    basis: list[int] = []
    for j in range(n):
        col_mask = vector_to_int(A[:, j])
        indep, basis = add_to_basis_rref(col_mask, basis)
        if not indep:
            return False
    return True


def matrix_to_fullrank_index(A: np.ndarray) -> int:
    """Inverse mapping: full-rank matrix -> payload index (raises if singular)."""
    A = np.array(A, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix must be square"

    basis: list[int] = []
    cols: list[int] = []
    for j in range(n):
        col_mask = vector_to_int(A[:, j])
        cols.append(col_mask)
        indep, basis = add_to_basis_rref(col_mask, basis)
        if not indep:
            raise ValueError("Matrix is not full-rank over GF(2)")

    return rank_fullrank_matrix(n, cols)


# ---------- Producer ----------


class FullrankMatrixProducer(MatrixProducer):
    """
    Same generator logic as MatrixProducer, but payload indexes only full-rank A.
    """

    def __init__(
        self,
        n: int,
        index: int,
        B_list: list[np.ndarray] | None = None,
        f0_idx: int | None = None,
        f1_idx: int | None = None,
        f_indices: list[int] | None = None,
        verbose: bool = False,
    ):
        assert 0 <= index < count_fullrank_matrices(
            n
        ), "index out of range for full-rank matrices"
        self.payload_index = index

        # Convert payload index (full-rank enumeration) to raw matrix bits index
        A = fullrank_index_to_matrix(n, index)
        matrix_bits_index = matrix_to_index(A)

        super().__init__(
            n,
            matrix_bits_index,
            B_list=B_list,
            f0_idx=f0_idx,
            f1_idx=f1_idx,
            f_indices=f_indices,
            verbose=verbose,
        )


# ---------- Recoverer ----------


class FullrankMatrixRecoverer(MatrixRecoverer):
    """
    MatrixRecoverer that post-processes recovered A to a full-rank payload index.
    """

    def __init__(
        self,
        n: int,
        B_list: list[np.ndarray] | None = None,
        f0_idx: int | None = None,
        f1_idx: int | None = None,
        f_indices: list[int] | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n,
            B_list=B_list,
            f0_idx=f0_idx,
            f1_idx=f1_idx,
            f_indices=f_indices,
            verbose=verbose,
        )
        self.payload_index: int | None = None

    def feed(self, data: np.ndarray | None) -> int | None:
        bit_index = super().feed(data)

        if self.payload_index is not None:
            return self.payload_index
        if bit_index is None:
            return None

        A = index_to_matrix(self.n, bit_index)
        if not _is_fullrank(A):
            if self.verbose:
                print("[Recoverer] Recovered singular A; waiting for more data")
            # Allow the base class to keep collecting transitions
            self.recovered_index = None
            return None

        self.payload_index = matrix_to_fullrank_index(A)
        return self.payload_index


# ---------- constructors for external benchmark harness ----------


override_D = lambda n: count_fullrank_matrices(n)


def producer_constructor(index: int, n: int, d: int) -> Producer:
    assert d == count_fullrank_matrices(n)
    mats, f_indices = _init_global_defaults(n)
    return FullrankMatrixProducer(n, index, B_list=mats, f_indices=f_indices)


def recoverer_constructor(n: int, d: int) -> Recoverer:
    assert d == count_fullrank_matrices(n)
    mats, f_indices = _init_global_defaults(n)
    return FullrankMatrixRecoverer(n, B_list=mats, f_indices=f_indices)
