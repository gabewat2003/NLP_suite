import numpy as np
import pytest

from word2vec.model import SGNS, loss_and_grads, scatter_add
from word2vec.pairs import doc_pairs
from word2vec.sampler import NegativeSampler
from word2vec.train import train
from word2vec.vocab import Vocab, tokenize


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + eps; fp = f()
        x[idx] = old - eps; fm = f()
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def test_gradient_check():
    rng = np.random.default_rng(0)
    V, d, K = 6, 4, 3
    W_in, W_out = rng.normal(size=(V, d)), rng.normal(size=(V, d))
    c, o, n = np.array([1]), np.array([2]), np.array([[3, 4, 5]])
    f = lambda: loss_and_grads(W_in, W_out, c, o, n)[0].sum()
    _, gv, gup, gun = loss_and_grads(W_in, W_out, c, o, n)
    num_in = numeric_grad(f, W_in)
    num_out = numeric_grad(f, W_out)
    assert np.allclose(num_in[1], gv[0], atol=1e-6)
    assert np.allclose(num_out[2], gup[0], atol=1e-6)
    for k, w in enumerate([3, 4, 5]):
        assert np.allclose(num_out[w], gun[0, k], atol=1e-6)


def test_gradient_check_repeated_ids():
    """Negatives that repeat (and collide with the center) still sum correctly."""
    rng = np.random.default_rng(1)
    V, d = 5, 3
    W_in, W_out = rng.normal(size=(V, d)), rng.normal(size=(V, d))
    c, o, n = np.array([0, 0]), np.array([1, 2]), np.array([[3, 3], [0, 4]])
    f = lambda: loss_and_grads(W_in, W_out, c, o, n)[0].sum()
    _, gv, gup, gun = loss_and_grads(W_in, W_out, c, o, n)
    g_in = np.zeros_like(W_in); np.add.at(g_in, c, gv)
    g_out = np.zeros_like(W_out); np.add.at(g_out, o, gup); np.add.at(g_out, n.ravel(), gun.reshape(-1, d))
    assert np.allclose(numeric_grad(f, W_in), g_in, atol=1e-6)
    assert np.allclose(numeric_grad(f, W_out), g_out, atol=1e-6)


def test_duplicate_indices_accumulate():
    rng = np.random.default_rng(0)
    m = SGNS(4, 3, rng)
    m.W_out = rng.normal(size=(4, 3))
    before = m.W_in.copy()
    c, o, n = np.array([0] * 5), np.array([1] * 5), np.array([[2]] * 5)
    _, gv, _, _ = loss_and_grads(m.W_in, m.W_out, c, o, n)
    m.step(c, o, n, lr=0.1)
    # all five updates to row 0 must land (fancy-index -= would apply only one)
    assert np.allclose(m.W_in[0], before[0] - 0.1 * gv.sum(0))


def test_sampler_distribution():
    counts = np.array([1000, 300, 100, 10, 1], dtype=float)
    s = NegativeSampler(counts, np.random.default_rng(0))
    draws = s.draw(1_000_000)
    emp = np.bincount(draws, minlength=5) / len(draws)
    want = counts ** 0.75 / (counts ** 0.75).sum()
    assert np.allclose(emp, want, atol=2e-3)


def test_sampler_never_returns_context():
    s = NegativeSampler(np.array([1000.0, 1.0]), np.random.default_rng(0))
    ctx = np.zeros(500, dtype=np.int64)
    assert not (s.sample(ctx, 5) == 0).any()


def test_doc_pairs_window():
    ids = np.arange(10)
    c, x = doc_pairs(ids, 3, np.random.default_rng(0))
    assert len(c) == len(x) > 0
    assert (np.abs(c - x) <= 3).all() and (c != x).all()
    c1, x1 = doc_pairs(np.array([7]), 3, np.random.default_rng(0))
    assert len(c1) == 0


def test_tokenize():
    assert tokenize("The charge-off on <DATE> wasn't mine!") == ["the", "charge-off", "on", "<DATE>", "wasn't", "mine"]


def _toy_corpus(rng, n=400):
    docs = []
    for _ in range(n):
        if rng.random() < 0.5:
            docs.append(["a", "b"] * 3 + list(rng.choice(["p", "q", "r"], 3)))
        else:
            docs.append(["c", "d"] * 3 + list(rng.choice(["s", "t", "u"], 3)))
    return docs


def test_learns_cooccurrence():
    emb, hist = train(_toy_corpus(np.random.default_rng(0)), dim=16, window=2, neg=3, min_count=1,
                      epochs=5, subsample=0, batch_size=32, seed=0, log_every=500)
    sim = lambda a, b: float(emb.vector(a) @ emb.vector(b))
    assert sim("a", "b") > sim("a", "c")
    assert hist[-1][1] < hist[0][1] * 0.7


def test_interchangeable_words_converge():
    rng = np.random.default_rng(0)
    docs = []
    for _ in range(600):
        w = "x" if rng.random() < 0.5 else "y"
        docs.append(["m", "n", w, "o", "p"])
        docs.append(["q", "r", "s", "t", "v"])
    emb, _ = train(docs, dim=16, window=2, neg=3, min_count=1, epochs=8, subsample=0,
                   batch_size=32, seed=0, log_every=1000)
    assert float(emb.vector("x") @ emb.vector("y")) > 0.9


def test_deterministic():
    docs = _toy_corpus(np.random.default_rng(1), 100)
    a, _ = train(docs, dim=8, window=2, min_count=1, epochs=1, seed=3, batch_size=16)
    b, _ = train(docs, dim=8, window=2, min_count=1, epochs=1, seed=3, batch_size=16)
    assert np.array_equal(a.matrix, b.matrix)


def test_vocab_keep_probs():
    v = Vocab(["the", "rare"], [1_000_000, 10])
    p = v.keep_probs(1e-4)
    assert p[0] < 0.05 and p[1] == 1.0


def test_scatter_add_matches_add_at():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 7, 200)
    vals = rng.normal(size=(200, 4))
    a = np.zeros((7, 4)); np.add.at(a, idx, vals)
    b = np.zeros((7, 4)); scatter_add(b, idx, vals)
    assert np.allclose(a, b)
