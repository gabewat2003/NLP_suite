"""Fetch CFPB complaint narratives and keep the fields this project uses.

The CFPB removed narratives from its public database in September 2026
(Consumer Complaint Database Release 24), so the official bulk export no
longer has text. This script uses a CC0 mirror captured before the removal:
Hugging Face dataset BEE-spoke-data/consumer-finance-complaints, "has-text"
subset, pinned to one revision. Each file is checked against the SHA-256 that
Hugging Face publishes for it.

Writes data/raw/complaints.parquet and data/raw/download_meta.json.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

import config

REPO = "BEE-spoke-data/consumer-finance-complaints"
REVISION = "088cc7308d4afc2a880f2329d08e2f7a09188ec6"
FILES = {  # path in repo -> published sha256 (LFS oid)
    "has-text/train-00000-of-00003.parquet": "60902c32f0a6b3db25a378570d421765d0a15ced396c0c471a9d60e49b38d833",
    "has-text/train-00001-of-00003.parquet": "367c06699e95a391916b4e73685112b3fe2d1242f9160654dcb995a73908a4e4",
    "has-text/train-00002-of-00003.parquet": None,  # filled from the HF API at download time
}
COLUMNS = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "Complaint ID": "complaint_id",
}


def url(path):
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"


def published_sha256(path):
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{path.rsplit('/', 1)[0]}"
    listing = json.loads(subprocess.check_output(["curl", "-sfL", api]))
    return next(f["lfs"]["oid"] for f in listing if f["path"] == path)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def fetch(path, dest, expected):
    if not dest.exists():
        print(f"Downloading {path}")
        subprocess.run(["curl", "-fL", "--retry", "3", "--progress-bar", "-o", str(dest), url(path)], check=True)
    got = sha256_file(dest)
    if got != expected:
        dest.unlink()
        raise SystemExit(f"checksum mismatch for {path}: {got} != {expected}")
    return got


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    config.RAW.mkdir(parents=True, exist_ok=True)
    parts, total, shas = [], 0, {}
    for path, expected in FILES.items():
        expected = expected or published_sha256(path)
        dest = config.RAW / path.replace("/", "__")
        shas[path] = fetch(path, dest, expected)
        df = pq.read_table(dest, columns=list(COLUMNS)).to_pandas()
        total += len(df)
        df = df.rename(columns=COLUMNS)
        df = df[df["narrative"].notna() & (df["narrative"].str.strip() != "")]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["date_received"] = pd.to_datetime(df["date_received"])
    df["complaint_id"] = df["complaint_id"].astype("int64")
    df = df.drop_duplicates(subset="complaint_id").sort_values("date_received", kind="stable")
    df.to_parquet(config.RAW / "complaints.parquet", index=False)
    meta = {
        "source": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION,
        "license": "CC0-1.0", "files_sha256": shas,
        "downloaded": datetime.now(timezone.utc).date().isoformat(),
        "total_rows": total, "rows_with_narrative": len(df),
        "date_min": str(df["date_received"].min().date()),
        "date_max": str(df["date_received"].max().date()),
    }
    (config.RAW / "download_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
