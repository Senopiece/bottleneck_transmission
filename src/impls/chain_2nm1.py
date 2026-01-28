import math
from typing import Set, Tuple

import numpy as np

from ._utils.conversions import (
    bool_array_to_uint16,
    make_message_vector,
    message_from_message_vector,
    uint16_to_bool_array,
)
from ._utils.fields import gf2nm1
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
    n, mask = gf2nm1.make_field(N)
    q = (1 << N) - 1

    # ==========================================================================
    # Sampler fabric
    # ==========================================================================
    def make_sampler(message: Message) -> Sampler:
        # Message vector is directly the polynomial coefficients
        # TODO: try encoding into permutation polynomials
        # TODO: try encoding into full-cycle permutation polynomials
        message_vector = make_message_vector(message, m, q)  # shape (m,)

        def f(x: np.uint16) -> np.uint16:
            return gf2nm1.evaluate_poly(x, message_vector, n, mask)

        all_states = set(np.uint16(i) for i in range(q))

        # ----------------------------------------------------------------------
        # Precompute tails = nodes with indegree 0 in the functional graph of f
        # (i.e., values not attained as f(x) for any x).
        # This is sampler-side only and does not depend on the channel.
        # ----------------------------------------------------------------------
        tails = set(all_states)
        for x in range(q):
            y = f(np.uint16(x))
            if y in tails:
                tails.discard(y)

        visited: Set[np.uint16] = set()

        # pick a new start state, prioritizing unvisited tails
        def reset() -> np.uint16:
            tails_unvisited = tails - visited
            if tails_unvisited:
                return next(iter(tails_unvisited))

            remaining = all_states - visited
            if remaining:
                return next(iter(remaining))

            visited.clear()
            candidates = tails

            return next(iter(candidates)) if candidates else np.uint16(0)

        curr = reset()

        while True:
            # yield, mark visited and compute next
            yield uint16_to_bool_array(curr, N)
            visited.add(curr)
            nxt = f(curr)

            # if next would repeat a visited state, break adjacency with a delimiter and jump
            if nxt in visited:
                yield uint16_to_bool_array(nxt, N)
                yield np.ones(N, dtype=np.bool_)  # delimiter
                curr = reset()
            else:
                curr = nxt

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

        message_vector = gf2nm1.interpolate_poly(outputs, inputs, n, mask)
        message = message_from_message_vector(message_vector, message_bitsize, q)

        return message

    return Protocol(
        make_sampler=make_sampler,
        make_estimator=make_estimator,
    )


def max_message_bitsize(packet_bitsize: int) -> int:
    if ispowprime_1_15(packet_bitsize):
        return floor_2n_m1_log2_2n_m1(packet_bitsize)
    return 0


def expected_packets_until_reconstructed(
    gilbert_eliott_k: Tuple[float, float, float, float],  # pGB, pBG, pG, pB
    packet_bitsize: int,
    message_bitsize: int,
) -> float:
    """
    Estimate E[number of transmitted packets until reconstruction]
    under a Gilbert-Elliott deletion channel + sampler-delimiters.

    Channel:
      - States: G (good), B (bad)
      - Transitions: P(G->B)=pGB, P(B->G)=pBG
      - Deletion probs: pG in G, pB in B
      - If not deleted, the packet is received.
    Sampler:
      - With probability lambda_delim a transmission is a delimiter (breaks adjacency).
      - Approximated via random-mapping renewal: lambda_delim = 1/(e+2),
        and P(two consecutive transmissions are both data) ≈ s_data_adj = 1 - 2*lambda_delim.

    Reconstruction model:
      - Estimator needs ~m adjacent received data-packet pairs (your prior approximation).
      - Uses the same closed-form approximation in terms of:
          P1 = P(a single transmission yields a received data packet)
          P2 = P(two consecutive transmissions yield two received data packets)

    Notes:
      - If pGB + pBG == 0, the chain has no unique stationary distribution (two absorbing states).
        This implementation assumes the process starts in G (piG=1, piB=0).
    """
    if packet_bitsize < 1:
        raise ValueError("packet_bitsize must be >= 1")
    if message_bitsize <= 0:
        return 0.0

    pGB, pBG, pG, pB = gilbert_eliott_k

    # Validate probabilities
    for name, v in [("pGB", pGB), ("pBG", pBG), ("pG", pG), ("pB", pB)]:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1].")

    n = packet_bitsize
    w = message_bitsize

    # m ≈ w / log2(2^n - 1)
    q = (1 << n) - 1  # 2^n - 1
    b = math.log(q, 2.0)
    m = w / b

    if m > q:
        return math.inf

    # Sampler-induced delimiter statistics
    s_data_adj = 1.0 - 2.0 / (math.e + 2.0)
    lambda_delim = 0.5 * (1.0 - s_data_adj)

    # Stationary distribution of the GE chain (if it exists)
    denom = pGB + pBG
    if denom > 0.0:
        piG = pBG / denom
        piB = pGB / denom
    else:
        # No transitions: absorbing; assume start in Good
        piG, piB = 1.0, 0.0

    # P1: one transmission is (data) and received
    #   = P(not delimiter) * E_state[ receive_prob(state) ]
    recv_marg = piG * pG + piB * pB
    P1 = (1.0 - lambda_delim) * recv_marg

    # P2: two consecutive transmissions are both (data) and both received
    #   = P(two are data) * P(both received over two steps of Markov chain)
    #
    # Markov two-step receive probability:
    #   sum_i pi_i * sum_j P(i->j) * r_i * r_j
    pGG = 1.0 - pGB
    pGB_ = pGB
    pBG_ = pBG
    pBB = 1.0 - pBG

    two_step_recv = piG * (pGG * (pG * pG) + pGB_ * (pG * pB)) + piB * (
        pBG_ * (pB * pG) + pBB * (pB * pB)
    )
    P2 = s_data_adj * two_step_recv

    # Sanity / degeneracy checks
    if P1 <= 0.0 or P2 <= 0.0:
        return math.inf

    # Prior closed form requires P1 > P2 to avoid negative "overhead" term.
    # With strong positive correlation, we can get P2 >= P1; in that case,
    # the formula is not applicable as-written, so return inf (signal "model mismatch").
    if P2 >= P1:
        return math.inf

    # Exact expectation for "need m adjacent successes"
    #   E[T] = m / P2 + (1 - P1) / (P1 - P2)
    overhead = (1.0 - P1) / (P1 - P2)
    return m / P2 + overhead
