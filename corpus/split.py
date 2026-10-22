"""Window -> normalize -> dedup -> time-based split -> cap.

    python -m corpus.split --labels        # print Product counts by year to choose the window
    python -m corpus.split                 # write data/processed/{train,val,test}.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from corpus.data import ids_hash
from corpus.dedup import dedup
from corpus.normalize import normalize


def label_table(df):
    t = df.assign(year=df["date_received"].dt.year).pivot_table(
        index="product", columns="year", values="complaint_id", aggfunc="count", fill_value=0)
    return t


def assign_splits(df, train_end, val_end):
    d = df["date_received"]
    split = np.where(d <= pd.Timestamp(train_end), "train",
                     np.where(d <= pd.Timestamp(val_end), "val", "test"))
    return pd.Series(split, index=df.index)


def cap(df, n, seed):
    """Uniform random sample of n rows (deterministic), returned in date order."""
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).sort_values(["date_received", "complaint_id"], kind="stable")


def build(raw_path, out_dir, window_start, train_end, val_end, window_end,
          max_train, max_eval, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(raw_path)
    n_raw = len(df)
    df = df[(df["date_received"] >= pd.Timestamp(window_start)) &
            (df["date_received"] <= pd.Timestamp(window_end))].copy()
    n_window = len(df)
    df["narrative"] = df["narrative"].map(normalize)
    df = df[df["narrative"].str.len() > 0]
    df, dstats = dedup(df)
    df["split"] = assign_splits(df, train_end, val_end)
    caps = {"train": max_train, "val": max_eval, "test": max_eval}
    meta = {"raw_rows": n_raw, "window_rows": n_window, "dedup": dstats,
            "window": [window_start, window_end], "train_end": train_end, "val_end": val_end,
            "seed": seed, "splits": {}}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name].drop(columns="split")
        before = len(part)
        part = cap(part, caps[name], seed).reset_index(drop=True)
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        meta["splits"][name] = {"rows_before_cap": before, "rows": len(part),
                                "ids_hash": ids_hash(part["complaint_id"]),
                                "date_min": str(part["date_received"].min().date()) if len(part) else None,
                                "date_max": str(part["date_received"].max().date()) if len(part) else None}
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(config.RAW / "complaints.parquet"))
    ap.add_argument("--out", default=str(config.PROCESSED))
    ap.add_argument("--labels", action="store_true", help="print Product counts per year and exit")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    if args.labels:
        df = pd.read_parquet(args.raw, columns=["date_received", "product", "complaint_id"])
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(label_table(df))
        return
    meta = build(args.raw, args.out, config.WINDOW_START, config.TRAIN_END, config.VAL_END,
                 config.WINDOW_END, config.MAX_TRAIN_DOCS, config.MAX_EVAL_DOCS, args.seed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
