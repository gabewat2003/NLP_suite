"""Shared pytest fixtures: the synthetic corpus run through the real split pipeline."""
import os
from pathlib import Path

import pytest

os.environ.setdefault("NLP_SUITE_DEVICE", "cpu")  # tests are deterministic on CPU; MPS kernels are not

FIXTURE_RAW = Path(__file__).parent / "tests" / "fixtures" / "mini_cfpb.parquet"
# Fixture dates span 2021-01 .. 2024-06; these boundaries give every split rows.
FIXTURE_SPLIT = dict(window_start="2021-01-01", train_end="2023-06-30",
                     val_end="2023-12-31", window_end="2024-06-30")


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory):
    from corpus.split import build

    out = tmp_path_factory.mktemp("processed")
    build(FIXTURE_RAW, out, max_train=None, max_eval=None, seed=0, **FIXTURE_SPLIT)
    return out
