"""BPE trainers. Both return the same ranked merge list for the same input.

Ties are broken by (count, pair): among equally frequent pairs the
lexicographically largest wins. Never by dict order, or runs can differ.
"""
import heapq
from collections import Counter, defaultdict


def pair_stats(wf):
    pairs = Counter()
    for word, f in wf.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += f
    return pairs


def merge_word(word, pair):
    a, b = pair
    out, i, n = [], 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_naive(wf, n_merges, max_seconds=None):
    """Reference trainer: recount every pair after every merge. Slow, simple, correct.

    max_seconds stops early (used only by the benchmark); the result is then a
    prefix of the full merge list.
    """
    import time
    deadline = None if max_seconds is None else time.perf_counter() + max_seconds
    wf = dict(wf)
    merges = []
    for _ in range(n_merges):
        if deadline is not None and time.perf_counter() > deadline:
            break
        pairs = pair_stats(wf)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        new = Counter()
        for w, f in wf.items():
            new[merge_word(w, best)] += f
        wf = new
    return merges


class _Key:
    """Heap key: higher count first, then lexicographically larger pair first."""
    __slots__ = ("count", "pair")

    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def train_incremental(wf, n_merges, progress=None):
    """Fast trainer: keep pair counts and a pair -> word-ids index, and after
    each merge update only the words that contain the merged pair.

    Per-merge cost is proportional to the affected words, not the whole table.
    The best pair comes from a heap with lazy invalidation: stale entries
    (whose count no longer matches) are skipped when popped.
    """
    words = [list(w) for w in wf]
    freqs = list(wf.values())
    counts = defaultdict(int)
    where = defaultdict(set)
    for i, w in enumerate(words):
        f = freqs[i]
        for p in zip(w, w[1:]):
            counts[p] += f
            where[p].add(i)
    heap = [_Key(c, p) for p, c in counts.items() if c > 0]
    heapq.heapify(heap)

    merges = []
    while len(merges) < n_merges and heap:
        top = heapq.heappop(heap)
        if counts.get(top.pair, 0) != top.count or top.count <= 0:
            continue  # stale entry
        best = top.pair
        merges.append(best)
        if progress and len(merges) % progress == 0:
            print(f"  merge {len(merges):,}/{n_merges:,}  {best}  count={top.count:,}", flush=True)
        changed = set()
        for i in where.pop(best, ()):
            w = words[i]
            if len(w) < 2:
                continue
            new = merge_word(w, best)
            if len(new) == len(w):
                continue  # stale index entry: pair no longer in this word
            f = freqs[i]
            for p in zip(w, w[1:]):
                counts[p] -= f
                changed.add(p)
            for p in zip(new, new[1:]):
                counts[p] += f
                where[p].add(i)
                changed.add(p)
            words[i] = list(new)
        counts.pop(best, None)
        changed.discard(best)
        for p in changed:
            c = counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, _Key(c, p))
            else:
                counts.pop(p, None)
    return merges
