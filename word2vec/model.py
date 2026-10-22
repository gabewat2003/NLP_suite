"""Skip-gram with negative sampling in NumPy: parameters, loss/gradients, SGD step."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def loss_and_grads(W_in, W_out, centers, contexts, negs):
    """Mean-free (summed over the batch) SGNS loss and per-row gradients.

    L = -log s(u_o.v_c) - sum_k log s(-u_k.v_c)
    dL/du_o = (s_o - 1) v_c ;  dL/du_k = s_k v_c ;  dL/dv_c = (s_o - 1) u_o + sum_k s_k u_k
    """
    v = W_in[centers]                                   # (B, d)
    u_pos = W_out[contexts]                             # (B, d)
    u_neg = W_out[negs]                                 # (B, K, d)
    s_pos = sigmoid((v * u_pos).sum(1))                 # (B,)
    s_neg = sigmoid(np.einsum("bd,bkd->bk", v, u_neg))  # (B, K)
    loss = -(np.log(s_pos + 1e-10) + np.log(1 - s_neg + 1e-10).sum(1))
    g_pos = s_pos - 1.0
    grad_v = g_pos[:, None] * u_pos + np.einsum("bk,bkd->bd", s_neg, u_neg)
    grad_u_pos = g_pos[:, None] * v
    grad_u_neg = s_neg[:, :, None] * v[:, None, :]
    return loss, grad_v, grad_u_pos, grad_u_neg


def scatter_add(W, idx, vals):
    """W[idx] += vals, correct when idx has duplicates.

    Same result as np.add.at(W, idx, vals), about 2.5x faster: a sparse 0/1
    matrix sums each run of equal ids in C, then one fancy-index update runs
    over ids that are now unique. (Plain W[idx] += vals would silently keep
    only one of the duplicate updates.)
    """
    uniq, inv = np.unique(idx, return_inverse=True)
    n = len(idx)
    S = sparse.csr_matrix((np.ones(n, dtype=vals.dtype), (inv, np.arange(n))), shape=(len(uniq), n))
    W[uniq] += S @ vals


class SGNS:
    def __init__(self, vocab_size, dim, rng, dtype=np.float64):
        self.W_in = ((rng.random((vocab_size, dim)) - 0.5) / dim).astype(dtype)  # center vectors v
        self.W_out = np.zeros((vocab_size, dim), dtype=dtype)                    # context vectors u

    def step(self, centers, contexts, negs, lr):
        """One SGD step. Gradients are summed over the batch (no /B), so lr keeps
        its per-pair meaning (0.025 as in the reference code).

        Updates go through scatter_add (an np.add.at equivalent), NOT W[idx] -= x:
        the same word id repeats within a batch and fancy-index assignment
        silently keeps only one of the duplicate updates.
        """
        loss, gv, gup, gun = loss_and_grads(self.W_in, self.W_out, centers, contexts, negs)
        d = self.W_in.shape[1]
        lr = self.W_in.dtype.type(lr)
        scatter_add(self.W_in, centers, -lr * gv)
        scatter_add(self.W_out, np.concatenate([contexts, negs.ravel()]),
                    -lr * np.concatenate([gup, gun.reshape(-1, d)]))
        return float(loss.mean())


class Embeddings:
    """Final vectors + lookup. For BPE units, a word's vector is the mean of its piece vectors."""

    def __init__(self, words, matrix, meta=None, tokenizer=None):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.unit = self.matrix / np.maximum(norms, 1e-8)
        self.meta = dict(meta or {})
        self.tokenizer = tokenizer

    def vector(self, word):
        if self.meta.get("units") == "bpe":
            ids = [self.index[p] for p in self.tokenizer.pieces(" " + word) if p in self.index]
            if not ids:
                return None
            v = self.unit[ids].mean(0)
            return v / max(np.linalg.norm(v), 1e-8)
        i = self.index.get(word)
        return None if i is None else self.unit[i]

    def __contains__(self, word):
        return self.vector(word) is not None

    def most_similar(self, word, topn=10, restrict=None):
        v = self.vector(word)
        if v is None:
            return []
        sims = self.unit @ v
        pool = np.argsort(-sims)
        out = []
        for i in pool:
            w = self.words[i]
            if w == word or (self.meta.get("units") == "bpe" and w.strip("▁") == word):
                continue
            if restrict is not None and not restrict(w):
                continue
            out.append((w, float(sims[i])))
            if len(out) == topn:
                break
        return out

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.matrix)
        (out / "vocab.json").write_text(json.dumps(self.words))
        (out / "config.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        return out

    @classmethod
    def load(cls, in_dir):
        d = Path(in_dir)
        meta = json.loads((d / "config.json").read_text())
        tok = None
        if meta.get("units") == "bpe":
            from bpe.tokenizer import Tokenizer
            tok = Tokenizer.load(meta["tokenizer"])
        return cls(json.loads((d / "vocab.json").read_text()), np.load(d / "embeddings.npy"), meta, tok)
