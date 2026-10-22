"""Write data/SOURCES.md from the download and split metadata."""
import json

import numpy as np
import pandas as pd

import config
from corpus.data import load_split


def main():
    dl = json.loads((config.RAW / "download_meta.json").read_text())
    sp = json.loads((config.PROCESSED / "split_meta.json").read_text())
    gen_path = config.PROCESSED / "general_meta.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else None
    df = pd.concat([load_split(s) for s in ("train", "val", "test")], ignore_index=True)
    words = df["narrative"].str.split().str.len()
    p50, p90, p99 = np.percentile(words, [50, 90, 99]).astype(int)
    lines = [
        "# Data sources",
        "",
        "## CFPB Consumer Complaint Database",
        "",
        "The CFPB removed complaint narratives from its public database in September 2026",
        "(Consumer Complaint Database Release 24), so the official export no longer has text.",
        "This project uses a CC0 mirror captured before the removal.",
        "",
        f"- Mirror: {dl['source']} (\"has-text\" subset), revision `{dl['revision']}`",
        f"- License: {dl['license']}",
        f"- Downloaded: {dl['downloaded']}; mirror covers {dl['date_min']} to {dl['date_max']}",
        "- File SHA-256s (checked against the values Hugging Face publishes):",
        *[f"  - `{k}`: `{v}`" for k, v in dl["files_sha256"].items()],
        f"- Total rows: {dl['total_rows']:,}",
        f"- Rows with a published narrative: {dl['rows_with_narrative']:,}",
        "",
        "## Processing",
        "",
        f"- Window: {sp['window'][0]} to {sp['window'][1]} ({sp['window_rows']:,} narratives)",
        f"- Exact duplicates removed: {sp['dedup']['exact_dupes_removed']:,}; "
        f"near duplicates (first-200-character key): {sp['dedup']['near_dupes_removed']:,}",
        f"- Split by Date received: train <= {sp['train_end']} < val <= {sp['val_end']} < test",
        f"- Caps (uniform sample, seed {sp['seed']}): train {config.MAX_TRAIN_DOCS:,}, "
        f"val/test {config.MAX_EVAL_DOCS:,}",
        "",
        "| Split | Rows before cap | Rows used | Dates |",
        "|---|---:|---:|---|",
    ]
    for name, s in sp["splits"].items():
        lines.append(f"| {name} | {s['rows_before_cap']:,} | {s['rows']:,} | {s['date_min']} to {s['date_max']} |")
    lines += [
        "",
        "Window choice: Product names were renamed or merged in April 2017 and again in 2023;",
        "between 2018 and 2022 every Product label is constant (`python -m corpus.split --labels`).",
        "",
        f"- Narrative length in words (all splits used): p50 {p50}, p90 {p90}, p99 {p99}",
        f"- Distinct Product values: {df['product'].nunique()}; distinct Issue values: {df['issue'].nunique()}",
        "",
    ]
    if gen:
        lines += [
            "## General-domain text (BPE comparison only)",
            "",
            f"- Source: {gen['source']} (Hugging Face `datasets`)",
            f"- Train sample: {gen['train_paragraphs']:,} paragraphs, {gen['train_bytes']:,} bytes "
            f"(target {gen['target_bytes']:,} = CFPB train size), seed {gen['seed']}",
            f"- Test: the WikiText-103 test split ({gen['test_paragraphs']:,} paragraphs)",
            "",
        ]
    (config.DATA / "SOURCES.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
