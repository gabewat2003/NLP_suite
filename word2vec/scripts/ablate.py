"""One-knob-at-a-time ablations from the base config, scored on VAL (macro-F1 of
the Product classifier on mean-pooled vectors). 3 seeds each, run in parallel.

To keep the grid affordable, ablations train on the first 50k train complaints
for 3 epochs; the chosen setting is retrained on the full train split by
train_w2v.py. Writes word2vec/artifacts/chosen_config.json.

    python -m word2vec.scripts.ablate [--workers 8]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import config
from eval.metrics_io import summarize, write_metrics
from word2vec.experiment import ART, BASE, classify, fit, load_tokens

SUBSET, EPOCHS = 50_000, 3
GRID = [("dim", 50), ("dim", 300), ("window", 2), ("window", 10), ("neg", 15),
        ("subsample", 1e-3), ("subsample", 1e-5), ("min_count", 2), ("min_count", 20),
        ("lowercase", False), ("vectors", "in+out"), ("units", "bpe")]


def one(args):
    knob, value, seed = args
    cfg = {"epochs": EPOCHS}
    if knob:
        cfg[knob] = value
    lower = cfg.get("lowercase", True)
    units = cfg.get("units", "word")
    tok = None
    tok_dir = str(config.ROOT / "bpe" / "artifacts" / "cfpb_8k") if units == "bpe" else None
    if units == "bpe":
        from bpe.tokenizer import Tokenizer
        tok = Tokenizer.load(tok_dir)
    tr_df, tr = load_tokens("train", lower, units, tok, limit=SUBSET)
    va_df, va = load_tokens("val", lower, units, tok)
    emb, _ = fit(cfg, seed, tr, tr_df["complaint_id"], tok_dir)
    s, _ = classify(emb, tr, tr_df["product"], va, va_df["product"], seed=seed)
    print(f"  {knob or 'base'}={value} seed={seed}: val macro-F1 {s['macro_f1']:.4f}", flush=True)
    return knob, value, seed, s["macro_f1"], emb.meta["train_seconds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(None, None, s) for s in config.SEEDS] + [(k, v, s) for k, v in GRID for s in config.SEEDS]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(one, jobs))
    rows = {}
    for knob, value, seed, f1, secs in results:
        rows.setdefault((knob, value), []).append(f1)
    table = [{"knob": k or "base", "value": v, **summarize(f)} for (k, v), f in rows.items()]
    base = next(r for r in table if r["knob"] == "base")
    # adopt a knob change only if its mean beats base by more than one base sd
    chosen = {}
    for r in table:
        if r["knob"] != "base" and r["mean"] > base["mean"] + max(base["sd"], 1e-9):
            prev = chosen.get(r["knob"])
            if prev is None or r["mean"] > prev[1]:
                chosen[r["knob"]] = (r["value"], r["mean"])
    cfg = {k: v for k, (v, _) in chosen.items()}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "chosen_config.json").write_text(json.dumps({"config": cfg, "base": BASE}, indent=2) + "\n")
    write_metrics("w2v_ablations", {"table": table, "chosen": cfg},
                  {"subset_docs": SUBSET, "epochs": EPOCHS, "seeds": list(config.SEEDS), "metric": "val macro-F1"})
    for r in sorted(table, key=lambda r: -r["mean"]):
        print(f"{r['knob']:>10} {str(r['value']):>8}  {r['mean']:.4f} ± {r['sd']:.4f}")
    print("chosen changes vs base:", cfg)


if __name__ == "__main__":
    main()
