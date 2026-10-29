"""TF-IDF (unigram + bigram) + logistic regression: the number every component is compared to.

Fit on train, C chosen on val, evaluated once on test. Two label sets:
Product (9 classes) and Product | Issue (rare labels collapsed to Other | Other).

    python -m eval.tfidf_baseline
"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import config
from corpus.data import ids_hash, load_split
from eval.classify import majority, scores, top_confusions
from eval.labels import label_space, labels
from eval.metrics_io import write_metrics

C_GRID = [0.3, 1.0, 3.0, 10.0]


def fit_vectorizer(train_texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True, dtype=np.float32,
                          lowercase=True, token_pattern=r"(?u)<[A-Z]+>|\b\w[\w'\-]*\b")
    return vec, vec.fit_transform(train_texts)


def run(task, train, val, test, Xtr, Xva, Xte, seed):
    keep = label_space(train) if task == "product_issue" else None
    ytr, yva, yte = (labels(d, task, keep) for d in (train, val, test))
    best = None
    # C ascending with warm starts: each fit starts from the previous (more regularized) solution,
    # which cuts the 56-class fit from hours to minutes. lbfgs can stop at 0 iterations from a warm
    # start (it happened at C=3, which then just repeated C=1's scores), so a 0-iteration fit is
    # redone from scratch: every C in the grid gets a real fit.
    clf = LogisticRegression(max_iter=1000, random_state=seed, warm_start=True)
    for C in C_GRID:
        clf.set_params(C=C)
        clf.fit(Xtr, ytr)
        if clf.n_iter_[0] == 0:
            print(f"  {task} C={C}: warm start did 0 iterations, refitting from scratch", flush=True)
            clf = LogisticRegression(max_iter=1000, random_state=seed, warm_start=True, C=C)
            clf.fit(Xtr, ytr)
        f1 = scores(yva, clf.predict(Xva))["macro_f1"]
        print(f"  {task} C={C}: val macro-F1 {f1:.4f}  ({clf.n_iter_[0]} iterations)", flush=True)
        if best is None or f1 > best[0]:
            import copy
            best = (f1, C, copy.deepcopy(clf))
    _, C, clf = best
    pred = clf.predict(Xte)
    out = {"chosen_C": C, "val_macro_f1": best[0], "test": scores(yte, pred),
           "majority_test": scores(yte, majority(ytr, len(yte))),
           "top_confusions": top_confusions(list(yte), list(pred)), "n_labels": int(ytr.nunique())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--task", nargs="+", default=["product", "product_issue"], choices=["product", "product_issue"])
    args = ap.parse_args()
    train, val, test = (load_split(s) for s in ("train", "val", "test"))
    vec, Xtr = fit_vectorizer(train["narrative"])
    Xva, Xte = vec.transform(val["narrative"]), vec.transform(test["narrative"])
    for task in args.task:
        res = run(task, train, val, test, Xtr, Xva, Xte, args.seed)
        write_metrics(f"tfidf_{task}", res, {"ngram": [1, 2], "min_df": 3, "max_features": 300_000,
                                             "C_grid": C_GRID, "fit_ids_hash": ids_hash(train["complaint_id"])})
        t = res["test"]
        print(f"{task}: test macro-F1 {t['macro_f1']:.4f}  acc {t['accuracy']:.4f}  "
              f"(majority macro-F1 {res['majority_test']['macro_f1']:.4f})")


if __name__ == "__main__":
    main()
