"""Downstream test on the TEST split: predict Product from mean-pooled vectors.

Rows: majority class, random vectors (same vocab/dim/pooling), my word2vec,
gensim word2vec (reference), TF-IDF + LR (from eval/tfidf_baseline.py).
Random, mine and gensim are 3 seeds each, mean ± sd.

    python -m word2vec.scripts.eval_classify
"""
import argparse

import config
from eval.classify import majority, scores, top_confusions
from eval.metrics_io import read_metrics, summarize, write_metrics
from word2vec.experiment import ART, classify, load_tokens, random_like
from word2vec.model import Embeddings


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    seeds = list(config.SEEDS)
    tr_df, tr = load_tokens("train")
    te_df, te = load_tokens("test")
    ytr, yte = tr_df["product"], te_df["product"]
    rows = {"majority": scores(list(yte), majority(ytr, len(yte)))}
    res = {"mine": [], "random": [], "gensim": []}
    per_class, confusions = None, None
    for s in seeds:
        emb = Embeddings.load(ART / f"seed{s}")
        sc, pred = classify(emb, tr, ytr, te, yte, seed=s)
        res["mine"].append(sc)
        if s == seeds[0]:
            per_class, confusions = sc["per_class_f1"], top_confusions(list(yte), list(pred))
        res["random"].append(classify(random_like(emb, 1000 + s), tr, ytr, te, yte, seed=s)[0])
        g = ART / f"gensim_seed{s}"
        if g.exists():
            res["gensim"].append(classify(Embeddings.load(g), tr, ytr, te, yte, seed=s)[0])
        print(f"  seed {s}: mine {res['mine'][-1]['macro_f1']:.4f}  random {res['random'][-1]['macro_f1']:.4f}",
              flush=True)
    for k, v in res.items():
        if v:
            rows[k] = {"macro_f1": summarize([x["macro_f1"] for x in v]),
                       "accuracy": summarize([x["accuracy"] for x in v])}
    tf = read_metrics("tfidf_product")
    if tf:
        rows["tfidf"] = {"macro_f1": tf["metrics"]["test"]["macro_f1"], "accuracy": tf["metrics"]["test"]["accuracy"]}
    write_metrics("w2v_classify", {"rows": rows, "mine_per_class_f1_seed0": per_class,
                                   "mine_top_confusions_seed0": confusions}, {"split": "test", "task": "product"})
    for k, v in rows.items():
        f = v["macro_f1"]
        print(f"{k:>9}: macro-F1 {f['mean']:.4f} ± {f['sd']:.4f}" if isinstance(f, dict) else f"{k:>9}: macro-F1 {f:.4f}")


if __name__ == "__main__":
    main()
