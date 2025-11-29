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
TODO: optimize the algorithms now for the execution time
TODO: mb relax single_binary_fullrank to some single_binary_smth to still make sure no degenerate payloads, but more payloads than fullrank matrices? - looks like no

- Investigate why fullrank appears to be better


Other ideas:
1. shuffle matrix
    A = D_{NxM}S_{MxM}, where S is a fullrank matrix, D is payload data matrix
    s.t. instead of Y = DX update we will have Y = AX => D = YX^-1S^-1
2. redundant matrix
    A = [D Z], where D is actual variable, and Z is fixed
    then Y = AX to Y = DX_u + ZX_l, where ZX_l is constant term and thus
    D = (Y + ZX_l)X_u^-1
    Therefore it does not require to collect more data than with just A = D,
    but shuffles the space better, hence maybe better convergence
3. 1 + 2 - the implementation that uses this shuffling is named with postfix r
    A = [D Z]S, S = [S_u S_l]^T, where Z is any, S_u invertible, S_l can be any too
    Y = AX
    Y = [D Z]SX
    U = SX, U = [U_u U_l]^T = [S_u S_l]^T X, U_u = S_u X, U_l = S_l X
    Y = [D Z]U
    Y = D U_u + Z U_l
    D = (Y + Z U_l) U_u^-1
    D = (Y + Z S_l X) X^-1 S_u^-1

    this shuffles the space twice coz why not