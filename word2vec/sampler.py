"""Negative sampler: unigram counts raised to the 3/4 power.

Rule: a negative that equals the true context word is redrawn (the reference
implementation skips such draws). A negative equal to the center is allowed.
"""
import numpy as np


class NegativeSampler:
    def __init__(self, counts, rng, power=0.75):
        p = np.asarray(counts, dtype=np.float64) ** power
        self.probs = p / p.sum()
        self.cdf = np.cumsum(self.probs)
        self.cdf[-1] = 1.0
        self.rng = rng

    def draw(self, shape):
        return np.searchsorted(self.cdf, self.rng.random(shape), side="right")

    def sample(self, contexts, k):
        negs = self.draw((len(contexts), k))
        bad = negs == contexts[:, None]
        while bad.any():
            negs[bad] = self.draw(int(bad.sum()))
            bad = negs == contexts[:, None]
        return negs
