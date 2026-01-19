from typing import Dict
import numpy as np


def majority_vote(counts: Dict[np.uint16, int]) -> np.uint16 | None:
    """
    Return the key with the highest count from the counts dictionary
    or None if empty or several keys have the same highest count.
    """
    if not counts:
        return None

    max_count = max(counts.values())

    best = None
    for k, v in counts.items():
        if v == max_count:
            if best is not None:
                return None
            best = k

    return best
