import json

import pandas as pd

from corpus.data import ids_hash, load_split
from corpus.dedup import dedup


def test_splits_disjoint_and_ordered(processed_dir):
    parts = {s: load_split(s, processed=processed_dir) for s in ("train", "val", "test")}
    for p in parts.values():
        assert len(p) > 0
    ids = [set(p["complaint_id"]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert parts["train"]["date_received"].max() < parts["val"]["date_received"].min()
    assert parts["val"]["date_received"].max() < parts["test"]["date_received"].min()


def test_meta_hash_matches(processed_dir):
    meta = json.loads((processed_dir / "split_meta.json").read_text())
    train = load_split("train", processed=processed_dir)
    assert meta["splits"]["train"]["ids_hash"] == ids_hash(train["complaint_id"])
    assert meta["dedup"]["exact_dupes_removed"] >= 2
    assert meta["dedup"]["near_dupes_removed"] >= 1


def test_no_raw_redactions_left(processed_dir):
    train = load_split("train", processed=processed_dir)
    assert not train["narrative"].str.contains("XXXX").any()
    assert not train["narrative"].str.contains(r"\{\$").any()


def test_dedup_keeps_earliest():
    df = pd.DataFrame({"narrative": ["same text here", "same text here"],
                       "date_received": pd.to_datetime(["2022-02-01", "2022-01-01"]),
                       "complaint_id": [2, 1]})
    out, stats = dedup(df)
    assert list(out["complaint_id"]) == [1]
    assert stats["exact_dupes_removed"] == 1
