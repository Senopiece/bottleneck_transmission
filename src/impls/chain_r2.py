from collections import defaultdict
import math
from typing import Dict

import random
import numpy as np

from ._utils.algo import majority_vote
from ._utils.conversions import (
    bool_array_to_uint16,
    make_rbackbone_vector,
    message_from_rbackbone_vector,
    uint16_to_bool_array,
)
from ._utils.gf2n import (
    berlekamp_welch,
    step,
    primitive_poly_int,
)
from ._interface import Config, Protocol, Message, Sampler, Estimator

# NOTE: this is the first steps outta domain of full guarantee on correct recover, chasing faster convergence and further working on faulty channels

# Domain:
# deletion_probability: [0, 1)
# corruption_probability: smth
# deletion_observation: 1.0

# NOTE: this is directing a new approach - it does not guarantee correct reconstruction 100% of the time, but just some % of the time with hope to converge faster - also it is more suitable under the channel with errors


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

    m = math.ceil(message_bitsize / packet_bitsize)
    n = packet_bitsize
    poly = np.uint16(primitive_poly_int(n))  # includes x^n term
    mask = np.uint16((1 << n) - 1)  # keep n bits
    red = np.uint16(poly & mask)  # poly without x^n term (low n bits)

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # there dont need to convert message to int intermediate since the avaliable alphabet is perfectly a power of two
        backbone = make_rbackbone_vector(message, m, n)

        # coeffs is just a backbone
        coeffs = backbone

        # Yield samples
        Nstates = 1 << n
        tails = set(np.uint16(i) for i in range(1, Nstates))
        for x in range(1, Nstates):
            y = step(np.uint16(x), coeffs, m, n, mask, red)
            tails.discard(y)

        visited = set()
        all_states = set(np.uint16(i) for i in range(Nstates))
        curr = np.uint16(1)

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
                yield uint16_to_bool_array(curr, n)

                if len(visited) == Nstates:
                    visited.clear()

                reset()

            yield uint16_to_bool_array(curr, n)
            visited.add(curr)
            curr = step(curr, coeffs, m, n, mask, red)

    # ==========================================================================
    # Estimator fabric
    # ==========================================================================
    def make_estimator() -> Estimator:
        # TODO: a proper way to choose the redundancy amount to add
        k = m + 10  # oversampling to correct for resets
        ready_evaluation_examples = 0
        evaluation_examples: Dict[np.uint16, Dict[np.uint16, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        x: np.uint16 | None = None

        ## Collect mapping examples
        while ready_evaluation_examples < k:
            packet = yield ready_evaluation_examples / k

            # interrupt streak by delimiter
            if packet is None:
                x = None
                continue

            # record example
            y = bool_array_to_uint16(packet)
            if x is not None:
                was_count_as_ready = majority_vote(evaluation_examples[x]) is not None

                evaluation_examples[x][y] += 1

                now_is_ready = majority_vote(evaluation_examples[x]) is not None

                if was_count_as_ready and not now_is_ready:
                    ready_evaluation_examples -= 1
                elif not was_count_as_ready and now_is_ready:
                    ready_evaluation_examples += 1

            x = y

        # first preprocess evaluation examples majority vote on duplicates
        xs = np.empty(k, dtype=np.uint16)
        ys = np.empty(k, dtype=np.uint16)
        i = 0
        for x, counts in evaluation_examples.items():
            y = majority_vote(counts)
            if y is not None:
                xs[i] = x
                ys[i] = y
                i += 1

        # assert xs has no duplicates
        # assert len(xs) == len(ys) == ready_evaluation_examples == k

        ## Reconstruct message with Berlekamp–Welch
        # we are given k evaluations in form y_i = coeffs @ monomials(x_i)
        # so Berlekamp–Welch solves for coeffs
        coeffs, ok = berlekamp_welch(xs, ys, m, n, mask, red)
        if not ok:
            raise RuntimeError("Berlekamp-Welch failed to decode.")

        # reconstruct the message data
        backbone = coeffs

        # reverse the packing
        message = message_from_rbackbone_vector(backbone, message_bitsize, m, n)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    return int(packet_bitsize * (1 << packet_bitsize) * 0.6)
