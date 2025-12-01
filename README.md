# Bottleneck transmission 
This repo provides a solution to the problem of transferring data packages through small internet packages (below 400 bits). Below you can see the algorithm for that.

**Algorithm 1:**
Inputs:
    D - the data that you want to transfer
Algorithm:
1. Initialize matrix Y of dim nxm, which you want to relay
2. Initialize A = Y $X^{-1}$, where $X$ is some standard basis
3. Initialize new set of basis vectors W, s.t. $w_{i+1} = A w_i$, $w_0 = x_0$
4. Send col. vectors of W via the channel
5. Get W on the other end of the channel
6. Recreate A from W
7. Recreate Y as Y = AX


