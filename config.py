"""Single source of settings shared by every component.

Set NLP_SUITE_PROCESSED to point the pipeline at another processed-data
directory (tests use this to run everything on the small fixture).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = Path(os.environ.get("NLP_SUITE_PROCESSED", DATA / "processed"))
REPORTS = ROOT / "reports"
METRICS = REPORTS / "metrics"
FIGURES = REPORTS / "figures"

SEED = 0
SEEDS = (0, 1, 2)  # seeds used for every mean ± sd number
# Task B final runs (seq2seq and classification head) use 5 seeds: with 3, the seed spread was
# as large as the gap to the TF-IDF baseline.
TASKB_SEEDS = (0, 1, 2, 3, 4)

# Time window and split boundaries (Date received). Chosen in corpus/split.py
# from the label-stability table; see data/SOURCES.md for the reasoning.
# Product names were renamed/merged in April 2017 and again in 2023, so the
# window is 2018-2022, where every Product label is constant.
WINDOW_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VAL_END = "2022-06-30"
WINDOW_END = "2022-12-31"

MAX_TRAIN_DOCS = 150_000
MAX_EVAL_DOCS = 20_000  # cap for val and for test

# Special tokens; ids are assigned in this order, starting at 0.
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<REDACTED>", "<DATE>", "<AMT>"]
REDACTION_TOKENS = ["<REDACTED>", "<DATE>", "<AMT>"]

# Product | Issue labels with fewer train examples than this collapse to
# "Other | Other" (55 labels kept, 97.8% of train rows).
MIN_LABEL_COUNT = 200
