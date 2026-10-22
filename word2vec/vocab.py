"""Word tokens, vocabulary counted on train only, and frequent-word subsampling."""
import json
import sys

import numpy as np
import regex

# Redaction tokens stay whole; words keep internal hyphens/apostrophes (charge-off, don't).
TOKEN_RE = regex.compile(r"<[A-Z]+>|[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*")


def tokenize(text, lowercase=True):
    toks = TOKEN_RE.findall(text)
    if lowercase:
        toks = [t if t.startswith("<") else t.lower() for t in toks]
    return [sys.intern(t) for t in toks]  # one copy per distinct word: ~1.5 GB less for the full split


class Vocab:
    """Words sorted by frequency (id 0 = most frequent). Words under min_count
    are dropped from the training stream, not mapped to <unk>."""

    def __init__(self, words, counts):
        self.words = list(words)
        self.counts = np.asarray(counts, dtype=np.float64)
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def build(cls, token_lists, min_count=5):
        from collections import Counter
        c = Counter()
        for toks in token_lists:
            c.update(toks)
        items = sorted(((w, n) for w, n in c.items() if n >= min_count), key=lambda x: (-x[1], x[0]))
        return cls([w for w, _ in items], [n for _, n in items])

    def __len__(self):
        return len(self.words)

    def __contains__(self, w):
        return w in self.index

    def encode(self, tokens):
        idx = self.index
        return np.fromiter((idx[t] for t in tokens if t in idx), dtype=np.int64)

    def keep_probs(self, t):
        """P(keep) per word, word2vec reference-C formula:
        min(1, sqrt(t / f) + t / f), f = word count / total tokens.
        (The paper's 1 - sqrt(t/f) discard rule is close but not identical.)"""
        f = self.counts / self.counts.sum()
        r = t / f
        return np.minimum(1.0, np.sqrt(r) + r)

    def to_json(self):
        return {"words": self.words, "counts": self.counts.astype(int).tolist()}

    @classmethod
    def from_json(cls, d):
        return cls(d["words"], d["counts"])

    def save(self, path):
        path.write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(path.read_text()))
