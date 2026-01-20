from typing import Set, Tuple

import random
import numpy as np

from ._utils.conversions import (
    make_zbackbone_vector,
    message_from_zbackbone_vector,
    bits_to_int,
    bool_array_to_uint16,
    int_to_bits,
    uint16_to_bool_array,
)
from ._utils.gf2n import (
    gf2n_matvec_right,
    gf2n_solve_and_invert,
    monomials_gf2n,
    step,
    primitive_poly_int,
)
from ._interface import Config, Protocol, Message, Sampler, Estimator
from ._utils.intmath import (
    floor_2n_log2_2n_minus_1,
    min_m_such_that_2n_minus_1_pow_k_ge_2p,
)


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = config.packet_bitsize
    message_bitsize = config.message_bitsize

    if not (1 <= packet_bitsize <= 16):
        raise ValueError(
            "GF(2^packet_bitsize) polynomial book integrated only for packet_bitsize in [1,16]."
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
    poly = np.uint16(primitive_poly_int(n))  # includes x^n term
    mask = np.uint16((1 << n) - 1)  # keep n bits
    red = np.uint16(poly & mask)  # poly without x^n term (low n bits)

    # V = [h(1), h(2), ..., h(m)] stacked as columns, shape (m, m)
    V = np.empty((m, m), dtype=np.uint16)
    for i in range(1, m + 1):
        V[:, i - 1] = monomials_gf2n(np.uint16(i), m, n, mask, red)

    Vinv = None  # if a sender needs it, it can compute it

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        nonlocal Vinv

        # Convert message to backbone vector
        int_msg = bits_to_int(message)
        backbone = make_zbackbone_vector(int_msg, m, n)  # shape (m,)

        # Convert backbone to generator coefficients vector
        if Vinv is None:
            # solve coeffs@V = backbone for coeffs
            coeffs, Vinv, singular = gf2n_solve_and_invert(V, backbone, n, mask, red)
            assert not singular, "Undegeneracy matrix is singular!"
        else:
            # use cached inverse
            coeffs = gf2n_matvec_right(backbone, Vinv, n, mask, red)

        # Yield samples
        Nstates = 1 << n
        tails = set(np.uint16(i) for i in range(1, Nstates))
        for x in range(1, Nstates):
            y = step(np.uint16(x), coeffs, m, n, mask, red)
            tails.discard(y)

        visited = {np.uint16(0)}
        all_states = set(np.uint16(i) for i in range(Nstates))
        prev = np.uint16(0)
        curr = np.uint16(1)

        def reset():
            nonlocal prev, curr
            prev = np.uint16(0)
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
                if curr != 0 and prev != 0:
                    yield uint16_to_bool_array(curr, n)

                yield np.zeros(n, dtype=np.bool_)

                if len(visited) == Nstates:
                    visited = {np.uint16(0)}

                reset()

            yield uint16_to_bool_array(curr, n)
            visited.add(curr)
            prev = curr
            curr = step(curr, coeffs, m, n, mask, red)

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
            y = bool_array_to_uint16(packet)
            if x is not None:
                evaluation_examples.add((x, y))
            x = y

        ## Reconstruct message
        X = np.empty((m, m), dtype=np.uint16)
        Y = np.empty(m, dtype=np.uint16)
        for i, (x_val, y_val) in enumerate(evaluation_examples):
            X[:, i] = monomials_gf2n(x_val, m, n, mask, red)
            Y[i] = y_val

        # solve Y = coeffs@X
        coeffs, _, singular = gf2n_solve_and_invert(X, Y, n, mask, red)
        if singular:
            # This should not happen with correct protocol usage
            raise RuntimeError("Failed to reconstruct message: singular matrix.")

        # solve backbone = coeffs@V
        backbone = gf2n_matvec_right(coeffs, V, n, mask, red)
        int_msg = message_from_zbackbone_vector(backbone, n)
        message = int_to_bits(int_msg, message_bitsize)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    return floor_2n_log2_2n_minus_1(packet_bitsize) - 10
