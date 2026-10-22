"""Cross-check against gensim with the same hyperparameters. Vectors differ, so
compare behavior: mean top-10 neighbor overlap on the probe words, against the
seed-to-seed noise floor of each implementation.

    python -m word2vec.scripts.crosscheck_gensim
"""
import argparse
import itertools

import numpy as np

import config
from bpe.evaluate import load_terms
from corpus.data import ids_hash
from eval.metrics_io import write_metrics
from word2vec.experiment import ART, BASE, chosen_config, load_tokens
from word2vec.model import Embeddings


def overlap(a, b, words, k=10):
    vals = []
    for w in words:
        if w in a.index and w in b.index:
            na = {x for x, _ in a.most_similar(w, k)}
            nb = {x for x, _ in b.most_similar(w, k)}
            vals.append(len(na & nb) / k)
    return float(np.mean(vals)), len(vals)


def train_gensim(toks, cfg, seed, fit_ids):
    from gensim.models import Word2Vec
    m = Word2Vec(sentences=toks, sg=1, hs=0, negative=cfg["neg"], window=cfg["window"],
                 vector_size=cfg["dim"], min_count=cfg["min_count"], sample=cfg["subsample"],
                 epochs=cfg["epochs"], alpha=cfg["lr0"], workers=8, seed=seed)
    words = list(m.wv.index_to_key)
    mat = m.wv.vectors if cfg["vectors"] == "in" else m.wv.vectors + m.syn1neg
    return Embeddings(words, mat, {"units": "word", "impl": "gensim", "seed": seed,
                                   "fit_ids_hash": ids_hash(fit_ids)})


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = dict(BASE, **chosen_config())
    df, toks = load_tokens("train", cfg["lowercase"])
    probes = [t.lower() if cfg["lowercase"] else t for t in load_terms()]
    for s in config.SEEDS:
        if not (ART / f"gensim_seed{s}").exists():
            train_gensim(toks, cfg, s, df["complaint_id"]).save(ART / f"gensim_seed{s}")
            print(f"  gensim seed {s} trained", flush=True)
    mine = {s: Embeddings.load(ART / f"seed{s}") for s in config.SEEDS}
    gen = {s: Embeddings.load(ART / f"gensim_seed{s}") for s in config.SEEDS}
    pairs = list(itertools.combinations(config.SEEDS, 2))
    mine_vs_gensim = [overlap(mine[s], gen[s], probes)[0] for s in config.SEEDS]
    gensim_floor = [overlap(gen[a], gen[b], probes)[0] for a, b in pairs]
    mine_floor = [overlap(mine[a], mine[b], probes)[0] for a, b in pairs]
    n = overlap(mine[0], gen[0], probes)[1]
    res = {"mine_vs_gensim": float(np.mean(mine_vs_gensim)), "gensim_vs_gensim": float(np.mean(gensim_floor)),
           "mine_vs_mine": float(np.mean(mine_floor)), "n_probe_words": n}
    write_metrics("w2v_crosscheck", res, {"k": 10, **cfg})
    print(res)


if __name__ == "__main__":
    main()
