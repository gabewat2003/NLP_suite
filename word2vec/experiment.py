"""Shared pieces for the word2vec scripts: tokenized splits, doc vectors, the
Product classifier probe, and a training wrapper that saves artifacts."""
import json

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config
from corpus.data import ids_hash, load_split
from eval.classify import scores
from word2vec.model import Embeddings
from word2vec.train import train
from word2vec.vocab import tokenize

ART = config.ROOT / "word2vec" / "artifacts"
BASE = dict(dim=100, window=5, neg=5, min_count=5, epochs=5, subsample=1e-4, lr0=0.025,
            batch_size=1024, lowercase=True, vectors="in", units="word")


def load_tokens(split, lowercase=True, units="word", tokenizer=None, limit=None):
    df = load_split(split, columns=["complaint_id", "narrative", "product"])
    if limit:
        df = df.iloc[:limit]
    if units == "bpe":
        toks = [[tokenizer.token_text(i) for i in tokenizer.encode(t)] for t in df["narrative"]]
    else:
        toks = [tokenize(t, lowercase) for t in df["narrative"]]
    return df, toks


def fit(cfg, seed, tokens, fit_ids, tokenizer_dir=None):
    c = dict(BASE, **cfg)
    emb, hist = train(tokens, dim=c["dim"], window=c["window"], neg=c["neg"], min_count=c["min_count"],
                      epochs=c["epochs"], lr0=c["lr0"], subsample=c["subsample"],
                      batch_size=c["batch_size"], seed=seed, vectors=c["vectors"], dtype=np.float32,
                      meta={"units": c["units"], "lowercase": c["lowercase"], "tokenizer": tokenizer_dir,
                            "fit_ids_hash": ids_hash(fit_ids)})
    return emb, hist


def doc_matrix(emb, token_lists):
    """Mean of the vectors of each document's in-vocabulary tokens."""
    idx = emb.index
    rows, cols = [], []
    for r, toks in enumerate(token_lists):
        ids = [idx[t] for t in toks if t in idx]
        rows.extend([r] * len(ids)); cols.extend(ids)
    M = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(token_lists), len(emb.words)))
    counts = np.asarray(M.sum(1)).ravel()
    X = M @ emb.matrix
    return X / np.maximum(counts, 1)[:, None]


def classify(emb, train_toks, y_train, eval_toks, y_eval, seed=0, C=1.0):
    sc = StandardScaler()
    Xtr = sc.fit_transform(doc_matrix(emb, train_toks))
    Xev = sc.transform(doc_matrix(emb, eval_toks))
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xtr, y_train)
    pred = clf.predict(Xev)
    return scores(list(y_eval), list(pred)), pred


def random_like(emb, seed):
    rng = np.random.default_rng(seed)
    return Embeddings(emb.words, rng.normal(size=emb.matrix.shape).astype(np.float32), {"units": "random"})


def chosen_config():
    p = ART / "chosen_config.json"
    return json.loads(p.read_text())["config"] if p.exists() else {}
