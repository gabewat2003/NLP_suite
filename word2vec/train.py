"""SGNS training loop: subsampling, dynamic windows, negatives, linear lr decay."""
import time

import numpy as np

from word2vec.model import SGNS, Embeddings
from word2vec.pairs import batches
from word2vec.sampler import NegativeSampler
from word2vec.vocab import Vocab


def train(docs_tokens, dim=100, window=5, neg=5, min_count=5, epochs=5, lr0=0.025,
          subsample=1e-4, batch_size=256, seed=0, vectors="in", log_every=200_000, meta=None,
          dtype=np.float64):
    """docs_tokens: list of token lists (one per complaint, train split only).

    Returns (Embeddings, history) where history is a list of (tokens_seen, mean_loss).
    """
    rng = np.random.default_rng(seed)
    vocab = Vocab.build(docs_tokens, min_count=min_count)
    docs = [vocab.encode(t) for t in docs_tokens]
    keep = vocab.keep_probs(subsample) if subsample else None
    sampler = NegativeSampler(vocab.counts, rng)
    model = SGNS(len(vocab), dim, rng, dtype=dtype)
    total = epochs * sum(len(d) for d in docs)
    seen, history, run_loss, run_n, next_log = 0.0, [], 0.0, 0, log_every
    t0 = time.perf_counter()
    for ep in range(epochs):
        for centers, contexts, consumed in batches(docs, window, batch_size, rng, keep):
            lr = lr0 * max(1e-4, 1.0 - seen / total)
            negs = sampler.sample(contexts, neg)
            run_loss += model.step(centers, contexts, negs, lr) * len(centers)
            run_n += len(centers)
            seen += consumed
            if seen >= next_log:
                history.append((int(seen), run_loss / max(run_n, 1)))
                run_loss, run_n = 0.0, 0
                next_log += log_every
        if run_n:
            history.append((int(seen), run_loss / run_n))
            run_loss, run_n = 0.0, 0
        print(f"  epoch {ep + 1}/{epochs}  loss {history[-1][1]:.4f}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    matrix = model.W_in if vectors == "in" else model.W_in + model.W_out
    info = dict(meta or {}, dim=dim, window=window, neg=neg, min_count=min_count, epochs=epochs,
                lr0=lr0, subsample=subsample, batch_size=batch_size, seed=seed, vectors=vectors,
                dtype=np.dtype(dtype).name, vocab_size=len(vocab), train_tokens=int(total / epochs),
                train_seconds=round(time.perf_counter() - t0, 1))
    emb = Embeddings(vocab.words, matrix, info)
    emb.counts = vocab.counts
    return emb, history
