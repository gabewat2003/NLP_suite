"""General-domain comparison text for the BPE 2x2: a WikiText-103 sample whose
UTF-8 size matches the CFPB train narratives.

Writes data/processed/general_train.txt and general_test.txt (one paragraph per line).
"""
import argparse
import json

import numpy as np

import config
from corpus.data import load_split


def target_bytes(processed=None):
    df = load_split("train", columns=["narrative"], processed=processed)
    return int(df["narrative"].map(lambda s: len(s.encode("utf-8")) + 1).sum())


def _paragraphs(rows):
    for r in rows:
        t = r["text"].strip()
        if t and not t.startswith("="):  # skip section headings
            yield t


def sample_to_bytes(paragraphs, n_bytes, seed):
    paragraphs = list(paragraphs)
    order = np.random.default_rng(seed).permutation(len(paragraphs))
    out, total = [], 0
    for i in order:
        if total >= n_bytes:
            break
        p = paragraphs[i]
        out.append(p)
        total += len(p.encode("utf-8")) + 1
    return out, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    from datasets import load_dataset  # heavy import, only needed here

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    want = target_bytes()
    train, got = sample_to_bytes(_paragraphs(ds["train"]), want, args.seed)
    test = list(_paragraphs(ds["test"]))
    (config.PROCESSED / "general_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (config.PROCESSED / "general_test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    meta = {"source": "Salesforce/wikitext wikitext-103-raw-v1", "target_bytes": want,
            "train_bytes": got, "train_paragraphs": len(train), "test_paragraphs": len(test),
            "seed": args.seed}
    (config.PROCESSED / "general_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
