"""Shared classifier evaluation: macro-F1, accuracy, per-class F1, confusion pairs."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def scores(y_true, y_pred):
    labels = sorted(set(y_true))
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "per_class_f1": {l: float(f) for l, f in zip(labels, per)}}


def top_confusions(y_true, y_pred, n=5):
    from collections import Counter
    c = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [{"true": t, "pred": p, "count": k} for (t, p), k in c.most_common(n)]


def majority(y_train, n):
    vals, counts = np.unique(np.asarray(y_train), return_counts=True)
    return [vals[counts.argmax()]] * n
