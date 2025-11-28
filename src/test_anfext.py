# anfext_test.py

import random
from impls.anfext import (
    nonlinear_producer,
    NonlinearRecoverer,
)
from itertools import islice


def run_test(n, m, steps=100):
    print("============================================")
    print(f"TEST n={n}  m={m}")
    print("============================================")

    # d = 2^(n*m)
    d = 2 ** (n * m)

    # choose random A index
    index = random.randrange(d)
    print(f"Random payload index = {index}")

    # Create producer + recoverer
    prod = nonlinear_producer(n, m, index, d, verbose=True)
    recv = NonlinearRecoverer(n, m, verbose=True)

    # Iterate producer
    for step, pkt in enumerate(islice(prod, steps)):
        print(f"\n==== Step {step} ====")
        print("Packet:", pkt)

        out = recv.feed(pkt)

        if out is not None:
            print("\n==========================")
            print("Recovered A index =", out)
            print("==========================")
            return out

    print("\n!!! FAILED TO RECOVER IN GIVEN STEPS !!!")
    return None


if __name__ == "__main__":
    run_test(n=5, m=32, steps=1000)
