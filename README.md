TODO: investigate if we use primitive matrices will it transmit data faster?
    (i) determine how to int <-> primitive matrix
    (ii) utilize the same generation idea
    (iii) make a sophisticated recoverer that is utilizing information that it will recover a primitive matrix, hence is faster than general full-rank matrix recovery
    (iv) investigate is there a way to make better generation
TODO: akin to primitive matrices, investigate using other special classes of full-rank matrices that have nice properties (e.g. circulant matrices, symmetric matrices, unitary matrices, orthogonal matrices, upper/lower triangular matrices, permutation matrices, positive definite, positive semi-definite/definite, lowrank but consists of multiplication of smaller fullrank matrices (through svd), normal matrices, hankel matrices, toeplitz matrices, matrices with some entries fixed to 0/1, etc...)
TODO: generalize to be able to transfer arbitary amount of data
     - fat matrices to give more information
     - xor of specific matrix combinations
     - some known matrix entries to reduce the amount of information to transfer
     - resulting vector is concat of results of multiple smaller matrices (e.g. 3x3 and 2x2) to reduce the amount of information to transfer
TODO: investigate is there a way to make sequences that they provide linearly independent vectors faster? like maybe insert more zeros, but choose such a patches that there are more linearly independent vectors?

- Investigate why fullrank appears to be better