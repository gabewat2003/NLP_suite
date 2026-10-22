"""Qualitative probes, made measurable.

1. Top-5 neighbors for 20 finance probe words from my model and gensim, written
   to a blind scoring sheet (source hidden, rows shuffled). After scoring,
   `--score` computes precision@5 per model.
2. Polysemy: top-10 neighbors of charge / balance / dispute / interest.
3. Subword link: misspell probe terms (one adjacent swap), keep those with no
   word-level vector, and check whether the BPE-unit model puts the correct
   spelling in its top-5 (word-level gets no vector at all).

    python -m word2vec.scripts.neighbors            # writes sheets + metrics
    python -m word2vec.scripts.neighbors --score    # after filling the 'sensible' column
"""
import argparse
import csv
import json

import numpy as np

import config
from bpe.evaluate import load_terms
from eval.metrics_io import read_metrics, write_metrics
from word2vec.experiment import ART
from word2vec.model import Embeddings

PROBES = ["garnish", "repossess", "autopay", "collections", "escrow", "forbearance", "garnishment",
          "repossession", "delinquent", "chargeback", "charge-off", "reinvestigation", "fcra", "fdcpa",
          "overdraft", "foreclosure", "servicer", "furnisher", "deferment", "arbitration"]
POLYSEMY = ["charge", "balance", "dispute", "interest"]
SHEET = config.REPORTS / "w2v_neighbor_sheet.csv"
KEY = config.REPORTS / "w2v_neighbor_key.json"


def misspell(w):
    """Swap the two middle characters (deterministic, keeps first/last letters)."""
    if len(w) < 5:
        return None
    i = len(w) // 2
    s = w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
    return s if s != w else None


def bpe_word_matrix(bpe, words):
    vecs = np.stack([bpe.vector(w) if bpe.vector(w) is not None else np.zeros(bpe.unit.shape[1]) for w in words])
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)


def subword_test(word_emb, bpe_emb, terms):
    vocab_words = [w for w in word_emb.words[:30000] if w.isalpha()]
    M = bpe_word_matrix(bpe_emb, vocab_words)
    rows = []
    for t in terms:
        t = t.lower()
        m = misspell(t)
        if not m or not t.isalpha() or m in word_emb.index or t not in vocab_words:
            continue
        v = bpe_emb.vector(m)
        top = [vocab_words[i] for i in np.argsort(-(M @ v))[:5]]
        rows.append({"correct": t, "misspelled": m, "word_level_has_vector": False,
                     "bpe_top5": top, "hit": t in top})
    return rows


def write_sheet(mine, gen, rng):
    items, key = [], {}
    for src, emb in (("mine", mine), ("gensim", gen)):
        for p in PROBES:
            for w, _ in emb.most_similar(p, 5):
                items.append((p, w, src))
    order = rng.permutation(len(items))
    with open(SHEET, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["item", "probe", "neighbor", "sensible"])
        for n, i in enumerate(order):
            p, w, src = items[i]
            wr.writerow([n, p, w, ""])
            key[n] = src
    KEY.write_text(json.dumps(key))


def score():
    key = {int(k): v for k, v in json.loads(KEY.read_text()).items()}
    hits = {"mine": [], "gensim": []}
    with open(SHEET) as f:
        for r in csv.DictReader(f):
            if r["sensible"].strip() == "":
                raise SystemExit(f"item {r['item']} not scored yet")
            hits[key[int(r["item"])]].append(int(r["sensible"]))
    prev = read_metrics("w2v_neighbors")["metrics"]
    prev["precision_at_5"] = {k: float(np.mean(v)) for k, v in hits.items()}
    write_metrics("w2v_neighbors", prev, {"probes": PROBES})
    print(prev["precision_at_5"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score:
        return score()
    mine, gen = Embeddings.load(ART / "seed0"), Embeddings.load(ART / "gensim_seed0")
    bpe = Embeddings.load(ART / "bpe_seed0")
    missing = [p for p in PROBES if p not in mine.index]
    write_sheet(mine, gen, np.random.default_rng(config.SEED))
    out = {
        "neighbors_mine": {p: mine.most_similar(p, 5) for p in PROBES},
        "neighbors_gensim": {p: gen.most_similar(p, 5) for p in PROBES},
        "polysemy": {w: mine.most_similar(w, 10) for w in POLYSEMY},
        "subword": subword_test(mine, bpe, load_terms()),
        "missing_probes": missing,
    }
    sw = out["subword"]
    out["subword_hit_at_5"] = float(np.mean([r["hit"] for r in sw])) if sw else None
    write_metrics("w2v_neighbors", out, {"probes": PROBES, "polysemy": POLYSEMY})
    for w, ns in out["polysemy"].items():
        print(w, [n for n, _ in ns])
    print("subword hit@5:", out["subword_hit_at_5"], f"over {len(sw)} misspellings")
    print(f"scoring sheet: {SHEET}")


if __name__ == "__main__":
    main()
