"""Exact match, character accuracy, and per-bucket breakdowns."""
from collections import defaultdict


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def exact_match(preds, golds):
    return sum(p == g for p, g in zip(preds, golds)) / max(len(golds), 1)


def char_accuracy(preds, golds):
    """1 - normalized edit distance, averaged (secondary metric)."""
    vals = [1 - edit_distance(p, g) / max(len(p), len(g), 1) for p, g in zip(preds, golds)]
    return sum(vals) / max(len(vals), 1)


def by_bucket(preds, golds, buckets):
    acc = defaultdict(list)
    for p, g, b in zip(preds, golds, buckets):
        acc[b].append(p == g)
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}
