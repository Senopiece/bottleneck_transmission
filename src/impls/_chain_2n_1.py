from typing import Set, Tuple

import random
import numpy as np

from ._utils.conversions import (
    make_backbone_vector,
    message_from_backbone_vector,
    bits_to_int,
    bool_array_to_uint16,
    int_to_bits,
    uint16_to_bool_array,
)
from ._utils.gf2n_1 import (
    gf2n_1_solve_and_invert,
    monomials_gf2n_1,
    step,
)
from ._interface import Config, Protocol, Message, Sampler, Estimator
from ._utils.intmath import (
    floor_2n_log2_2n_minus_1,
    ispowprime_1_15,
    min_m_such_that_2n_minus_1_pow_k_ge_2p,
)

# TODO: fix, now not working

# TODO: try to generalize this method onto ispowprime_1_15=False cases - by using just Z/(2^n - 1)Z ring
#       the only problem is solving linear equations in that ring - which is not always possible
#       because of two reasons:
#       1) zero divisors during elimination
#       2) Vandermonde matrix determinant is not guaranteed to be nonzero
#
#       - but there are still matrices that are invertible even in that ring
#       so we just loose these two guarantees and need to write a more general solver that works in Z/(2^n - 1)Z
#
#       this can be done by collecting a wide X of shape mxk, k>=m, then for all possible m columns from this X, collect a X' that is a mxm matrix, check is it ivertible (e.g. hope while inverting X no zero divisors appear and the detetminant is not zero). if no invertible mxm submatrix found, we can just ask for more samples to make next k > prev k, then try again. seems like it is guaranteed to invert eventually (TODO: verify it). next TODO: every time taking more and more subsets is costly, so further come up with some faster algo that is euivalent to that but reuses computation.
# TODO: also generalization can be achieved by decomposition of the 2^n - 1 into prime factors, then running distinct protocols in parallel. powers of primes to be handled with polynomial expansion like in gf2n case, distinct primes send and receive in parallel / as mixed field linear system. but this is more complex and probably not worth the effort. - so it also lets us leave in GF(2^n - 1), but with costly preparations and field ops and much more checks in the code - so slower code.


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = config.packet_bitsize
    message_bitsize = config.message_bitsize

    if not (1 <= packet_bitsize <= 15):
        raise ValueError(
            "current implementation only supports packet_bitsize in [1, 15]"
        )  # due to np.uint16 usage
    if ispowprime_1_15(packet_bitsize) is False:
        raise ValueError(
            "packet_bitsize must be such that 2^n - 1 is prime for this protocol"
        )

    # ==========================================================================
    # Precomputations
    # ==========================================================================

    m = min_m_such_that_2n_minus_1_pow_k_ge_2p(
        message_bitsize,
        packet_bitsize,
        1 << packet_bitsize,
    )

    n = packet_bitsize
    mask = np.uint16((1 << n) - 1)  # keep n bits

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # Convert message to backbone vector
        int_msg = bits_to_int(message)
        backbone = make_backbone_vector(int_msg, m, n)  # shape (m,)

        # Convert backbone to generator coefficients vector
        coeffs = backbone  # ya, its simply that

        # Yield samples
        Nstates = (1 << n) - 1  # GF(2^n - 1) states
        tails = set(np.uint16(i) for i in range(Nstates))
        for x in range(Nstates):
            y = step(np.uint16(x), coeffs, m, n, mask)
            tails.discard(y)

        visited: Set[np.uint16] = set()
        all_states = set(np.uint16(i) for i in range(Nstates))
        curr: np.uint16 = np.uint16(1)

        def reset():
            nonlocal curr
            tails_unvisited = set(tails) - visited
            remaining = all_states - visited
            curr = (
                random.choice(list(tails_unvisited))
                if tails_unvisited
                else random.choice(list(remaining))
            )

        reset()

        while True:
            if curr in visited:
                yield uint16_to_bool_array(curr + 1, n)
                yield np.zeros(n, dtype=np.bool_)

                if len(visited) == Nstates:
                    visited.clear()

                reset()

            yield uint16_to_bool_array(curr + 1, n)
            visited.add(curr)
            curr = step(curr, coeffs, m, n, mask)

    # ==========================================================================
    # Estimator fabric
    # ==========================================================================
    def make_estimator() -> Estimator:
        evaluation_examples: Set[Tuple[np.uint16, np.uint16]] = set()
        x: np.uint16 | None = None

        ## Collect mapping examples
        while len(evaluation_examples) < m:
            packet = yield len(evaluation_examples) / m

            # interrupt streak by delimiter or reset indicator
            if packet is None or np.all(packet == 0):
                x = None
                continue

            # record example
            y = bool_array_to_uint16(packet) - 1
            if x is not None:
                evaluation_examples.add((x, y))
            x = y

        ## Reconstruct message
        X = np.empty((m, m), dtype=np.uint16)
        Y = np.empty(m, dtype=np.uint16)
        for i, (x_val, y_val) in enumerate(evaluation_examples):
            X[i, :] = monomials_gf2n_1(x_val, m, n, mask)
            Y[i] = y_val

        # solve Y = coeffs@X
        coeffs, _, singular = gf2n_1_solve_and_invert(X, Y, n, mask)
        if singular:
            # This should not happen with correct protocol usage
            raise RuntimeError("Failed to reconstruct message: singular matrix.")

        # backbone = coeffs
        backbone = coeffs
        int_msg = message_from_backbone_vector(backbone, n)
        message = int_to_bits(int_msg, message_bitsize)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if ispowprime_1_15(packet_bitsize):
        return floor_2n_log2_2n_minus_1(packet_bitsize) - 10
    return 0
