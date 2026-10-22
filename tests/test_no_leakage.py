"""Every fit step must have seen the train split only.

Each tokenizer, embedding and vectorizer records a hash of the complaint IDs it
was fit on. These tests check those hashes against the train split.
"""
import json

import pytest

import config
from corpus.data import ids_hash, load_split


def test_bpe_fit_corpus_is_train(processed_dir, monkeypatch):
    from bpe.scripts.train_bpe import load_corpus
    monkeypatch.setattr(config, "PROCESSED", processed_dir)
    _, meta = load_corpus("cfpb")
    train = load_split("train", processed=processed_dir)
    assert meta["corpus_hash"] == ids_hash(train["complaint_id"])
    for split in ("val", "test"):
        assert meta["corpus_hash"] != ids_hash(load_split(split, processed=processed_dir)["complaint_id"])


def test_w2v_fit_is_train(processed_dir, monkeypatch):
    from word2vec.experiment import fit, load_tokens
    monkeypatch.setattr(config, "PROCESSED", processed_dir)
    df, toks = load_tokens("train")
    emb, _ = fit({"epochs": 1, "dim": 8, "min_count": 1}, 0, toks, df["complaint_id"])
    assert emb.meta["fit_ids_hash"] == ids_hash(df["complaint_id"])
    test_vocab = {t for toks in load_tokens("test")[1] for t in toks}
    train_vocab = {t for t_ in toks for t in t_}
    for w in test_vocab - train_vocab:
        assert w not in emb.index  # test-only words get no vector


def _real_train_hash():
    meta = config.DATA / "processed" / "split_meta.json"
    if not meta.exists():
        pytest.skip("real data not built (make data)")
    return json.loads(meta.read_text())["splits"]["train"]["ids_hash"]


def _artifact_hashes():
    out = []
    for p in (config.ROOT / "bpe" / "artifacts").glob("cfpb_*/config.json"):
        out.append((p, json.loads(p.read_text()).get("corpus_hash")))
    for p in (config.ROOT / "word2vec" / "artifacts").glob("*/config.json"):
        out.append((p, json.loads(p.read_text()).get("fit_ids_hash")))
    for p in config.METRICS.glob("*.json"):
        h = json.loads(p.read_text()).get("config", {}).get("fit_ids_hash")
        if h:
            out.append((p, h))
    return out


def test_real_artifacts_fit_on_train_only():
    want = _real_train_hash()
    found = _artifact_hashes()
    if not found:
        pytest.skip("no trained artifacts yet")
    for path, h in found:
        assert h == want, f"{path} was fit on something other than the train split"
