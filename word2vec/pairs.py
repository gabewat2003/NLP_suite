"""(center, context) pairs with a dynamic window, generated lazily in batches.

Windows never cross document (complaint) boundaries. For each position a radius
b ~ Uniform{1..window} is drawn, so nearer words are paired more often.
"""
import numpy as np


def doc_pairs(ids, window, rng):
    """All pairs for one document, vectorized. Returns (centers, contexts)."""
    n = len(ids)
    if n < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    b = rng.integers(1, window + 1, size=n)
    pos = np.arange(n)
    cs, xs = [], []
    for o in range(1, min(window, n - 1) + 1):
        m = b[: n - o] >= o              # center i looks right to i+o
        cs.append(ids[: n - o][m]); xs.append(ids[o:][m])
        m = b[o:] >= o                   # center i looks left to i-o
        cs.append(ids[o:][m]); xs.append(ids[: n - o][m])
    del pos
    return np.concatenate(cs), np.concatenate(xs)


def batches(docs, window, batch_size, rng, keep_probs=None, chunk_docs=2000):
    """Yield (centers, contexts, n_tokens_consumed) over one pass of docs in random order.

    keep_probs: per-word keep probability for subsampling (applied per occurrence,
    fresh each epoch). Pairs are shuffled within chunks of documents.
    """
    order = rng.permutation(len(docs))
    buf_c, buf_x = [], []
    for start in range(0, len(order), chunk_docs):
        consumed = 0
        cs, xs = [], []
        for d in order[start:start + chunk_docs]:
            ids = docs[d]
            consumed += len(ids)
            if keep_probs is not None and len(ids):
                ids = ids[rng.random(len(ids)) < keep_probs[ids]]
            c, x = doc_pairs(ids, window, rng)
            cs.append(c); xs.append(x)
        if not cs:
            continue
        c = np.concatenate(cs); x = np.concatenate(xs)
        if buf_c:
            c = np.concatenate([buf_c[0], c]); x = np.concatenate([buf_x[0], x])
            buf_c, buf_x = [], []
        p = rng.permutation(len(c))
        c, x = c[p], x[p]
        n_full = len(c) // batch_size
        per = consumed / max(n_full, 1)
        for i in range(n_full):
            s = slice(i * batch_size, (i + 1) * batch_size)
            yield c[s], x[s], per
        rest = slice(n_full * batch_size, None)
        if len(c[rest]):
            buf_c, buf_x = [c[rest]], [x[rest]]
    if buf_c:
        yield buf_c[0], buf_x[0], 0
