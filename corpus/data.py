"""Helpers for reading the processed splits and fingerprinting what a fit step saw."""
import hashlib

import pandas as pd

import config

SPLITS = ("train", "val", "test")


def split_path(split, processed=None):
    return (processed or config.PROCESSED) / f"{split}.parquet"


def load_split(split, columns=None, processed=None):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return pd.read_parquet(split_path(split, processed), columns=columns)


def ids_hash(complaint_ids):
    """Order-independent fingerprint of a set of complaint IDs.

    Every fit step stores this; tests/test_no_leakage.py checks it equals the
    train split's hash.
    """
    h = hashlib.sha256()
    for cid in sorted(str(c) for c in complaint_ids):
        h.update(cid.encode())
        h.update(b"\n")
    return h.hexdigest()


def texts(split, processed=None, limit=None):
    df = load_split(split, columns=["complaint_id", "narrative"], processed=processed)
    if limit is not None:
        df = df.iloc[:limit]
    return df
