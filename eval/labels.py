"""Label sets shared by the classifiers and seq2seq Task B."""
import config

OTHER = "Other | Other"


def product_issue(df):
    return df["product"] + " | " + df["issue"]


def label_space(train_df, min_count=None):
    """Product | Issue labels kept (>= min_count train rows), fit on train only."""
    min_count = config.MIN_LABEL_COUNT if min_count is None else min_count
    vc = product_issue(train_df).value_counts()
    return sorted(vc[vc >= min_count].index)


def collapse(series, keep):
    keep = set(keep)
    return series.where(series.isin(keep), OTHER)


def labels(df, task, keep=None):
    if task == "product":
        return df["product"]
    if task == "product_issue":
        return collapse(product_issue(df), keep)
    raise ValueError(task)
