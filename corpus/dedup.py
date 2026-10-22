"""Drop exact and near-duplicate narratives (template letters submitted many times)."""
import re

import pandas as pd

_KEY_RE = re.compile(r"[^a-z0-9]+")


def near_key(text, n=200):
    """Lowercased first n characters with punctuation and spaces removed."""
    return _KEY_RE.sub("", text.lower()[: n * 2])[:n]


def dedup(df, text_col="narrative", date_col="date_received"):
    """Keep the earliest complaint for each exact text and each near-duplicate key.

    Returns (deduped_df, stats).
    """
    n0 = len(df)
    df = df.sort_values([date_col, "complaint_id"], kind="stable")
    df = df.drop_duplicates(subset=[text_col], keep="first")
    n1 = len(df)
    keys = df[text_col].map(near_key)
    df = df[~keys.duplicated(keep="first")]
    n2 = len(df)
    stats = {"input": n0, "exact_dupes_removed": n0 - n1, "near_dupes_removed": n1 - n2, "kept": n2}
    return df.reset_index(drop=True), stats


__all__ = ["dedup", "near_key", "pd"]
