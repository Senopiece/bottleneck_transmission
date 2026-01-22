import math
from typing import Set, Tuple

import numpy as np

from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.fields.gf2nm1 import (
    evaluate_poly,
    interpolate_poly,
    make_feld,
)
from ._interface import Config, Protocol, Message, Sampler, Estimator
from ._utils.intmath import (
    ispowprime_1_15,
    min_m_such_that_2n_minus_1_pow_k_ge_2p,
    floor_2n_m1_log2_2n_m1,
)

# Domain:
# deletion_probability: [0, 1)
# corruption_probability: 0
# deletion_observation: 1.0


def create_protocol(config: Config) -> Protocol:
    packet_bitsize = config.packet_bitsize
    message_bitsize = config.message_bitsize

    # ==========================================================================
    # Precomputations
    # ==========================================================================

    N = packet_bitsize
    m = min_m_such_that_2n_minus_1_pow_k_ge_2p(
        message_bitsize,
        packet_bitsize,
        (1 << packet_bitsize) - 1,
    )
    n, mask = make_feld(N)

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # Message vector is directly the polynomial coefficients
        message_vector = make_message_vector(message, N, m)  # shape (m,)

        # Yield samples
        Nstates = (1 << N) - 1  # GF(2^n - 1) states
        tails = set(np.uint16(i) for i in range(Nstates))
        for x in range(Nstates):
            y = evaluate_poly(np.uint16(x), message_vector, n, mask)
            tails.discard(y)

        visited: Set[np.uint16] = set()
        all_states = set(np.uint16(i) for i in range(Nstates))
        curr: np.uint16 = np.uint16(1)

        def reset():
            nonlocal curr
            tails_unvisited = tails - visited
            remaining = all_states - visited
            curr = list(tails_unvisited)[0] if tails_unvisited else list(remaining)[0]

        reset()

        while True:
            if curr in visited:
                yield uint16_to_bool_array(curr, N)
                yield np.ones(N, dtype=np.bool_)

                if len(visited) == Nstates:
                    visited.clear()

                reset()

            yield uint16_to_bool_array(curr, N)
            visited.add(curr)
            curr = evaluate_poly(curr, message_vector, n, mask)

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
            if packet is None or np.all(packet == 1):
                x = None
                continue

            # record example
            y = bool_array_to_uint16(packet)
            if x is not None:
                evaluation_examples.add((x, y))
            x = y

        ## Reconstruct message
        outputs = np.empty(m, dtype=np.uint16)
        inputs = np.empty(m, dtype=np.uint16)

        for i, (x_val, y_val) in enumerate(evaluation_examples):
            inputs[i] = x_val
            outputs[i] = y_val

        message_vector = interpolate_poly(outputs, inputs, n, mask)
        message = message_from_message_vector(message_vector, N, message_bitsize)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if ispowprime_1_15(packet_bitsize):
        return floor_2n_m1_log2_2n_m1(packet_bitsize)
    return 0


def estimate_packets_until_reconstructed(
    deletion_prob: float, packet_bitsize: int, message_bitsize: int
) -> float:
    """
    Estimate E[number of transmitted packets until reconstruction]
    for the given protocol under independent deletions.

    Assumptions:
      - i.i.d. deletion channel with probability `deletion_prob`
      - sampler delimiter process approximated as a renewal process
        induced by a random functional graph on GF(2^N - 1)
      - duplicates among early (x,y) pairs are negligible up to m samples

    its tight for up to some inflection message_bitsize, after that coupon collector regime kicks in
    the more faulty channel is, the lower this inflection message_bitsize
    as we consider a low faulty channel, this becomes a tight estimate overall

    Parameters
    ----------
    deletion_prob : float
        Packet deletion probability d, 0 <= d < 1.
    packet_bitsize : int
        N, packet size in bits.
    message_bitsize : int
        Payload size in bits.

    Returns
    -------
    float
        Expected number of transmitted packets until message reconstruction.
    """
    if not (0.0 <= deletion_prob < 1.0):
        raise ValueError("deletion_prob must be in [0,1).")
    if packet_bitsize < 1:
        raise ValueError("packet_bitsize must be >= 1")
    if message_bitsize <= 0:
        return 0.0

    n = packet_bitsize
    w = message_bitsize

    # Same m as the protocol uses, but smooth one
    #
    # 2^w = (2^n - 1)^m
    # w = mlog2(2^n - 1)
    # m = w / log2(2^n - 1)

    q = (1 << n) - 1  # 2^n - 1
    b = math.log(q, 2)
    m = w / b

    if m > q:
        return math.inf

    # Sampler-induced delimiter statistics
    #
    # Random-mapping approximation:
    #   expected number of restart events per full traversal:
    #       L ≈ q / e
    #
    # delimiter fraction:
    #   λ = L / (q + 2L) = 1 / (e + 2)
    #
    # fraction of consecutive transmissions that are both data:
    #   s = 1 - 2λ = e / (e + 2)
    lambda_delim = 1.0 / (math.e + 2.0)
    s_data_adj = 1.0 - 2.0 * lambda_delim

    # Channel + sampler
    d = deletion_prob
    P1 = (1.0 - lambda_delim) * (
        1.0 - d
    )  # probability a single transmission yields a received data packet
    P2 = (
        s_data_adj * (1.0 - d) ** 2
    )  # probability two consecutive transmissions yield two received data packets

    if P2 <= 0.0 or P1 <= 0.0 or P1 <= P2:
        return math.inf

    # Exact expectation for "need m adjacent successes"
    #
    #   E[T] = m / P2 + (1 - P1) / (P1 - P2)
    overhead = (1.0 - P1) / (P1 - P2)
    return m / P2 + overhead
