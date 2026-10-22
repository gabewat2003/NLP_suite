# Data sources

## CFPB Consumer Complaint Database

The CFPB removed complaint narratives from its public database in September 2026
(Consumer Complaint Database Release 24), so the official export no longer has text.
This project uses a CC0 mirror captured before the removal.

- Mirror: https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints ("has-text" subset), revision `088cc7308d4afc2a880f2329d08e2f7a09188ec6`
- License: CC0-1.0
- Downloaded: 2026-10-22; mirror covers 2015-03-19 to 2024-02-09
- File SHA-256s (checked against the values Hugging Face publishes):
  - `has-text/train-00000-of-00003.parquet`: `60902c32f0a6b3db25a378570d421765d0a15ced396c0c471a9d60e49b38d833`
  - `has-text/train-00001-of-00003.parquet`: `367c06699e95a391916b4e73685112b3fe2d1242f9160654dcb995a73908a4e4`
  - `has-text/train-00002-of-00003.parquet`: `c9ad758d64795e780ef6b1f6a1367b7e18877db0a6be530a50777c2d205d8127`
- Total rows: 1,689,573
- Rows with a published narrative: 1,689,573

## Processing

- Window: 2018-01-01 to 2022-12-31 (958,430 narratives)
- Exact duplicates removed: 191,865; near duplicates (first-200-character key): 85,303
- Split by Date received: train <= 2021-12-31 < val <= 2022-06-30 < test
- Caps (uniform sample, seed 0): train 150,000, val/test 20,000

| Split | Rows before cap | Rows used | Dates |
|---|---:|---:|---|
| train | 482,157 | 150,000 | 2018-01-01 to 2021-12-31 |
| val | 91,908 | 20,000 | 2022-01-01 to 2022-06-30 |
| test | 107,197 | 20,000 | 2022-07-01 to 2022-12-31 |

Window choice: Product names were renamed or merged in April 2017 and again in 2023;
between 2018 and 2022 every Product label is constant (`python -m corpus.split --labels`).

- Narrative length in words (all splits used): p50 128, p90 402, p99 1076
- Distinct Product values: 9; distinct Issue values: 80

## General-domain text (BPE comparison only)

- Source: Salesforce/wikitext wikitext-103-raw-v1 (Hugging Face `datasets`)
- Train sample: 268,176 paragraphs, 165,008,697 bytes (target 165,007,197 = CFPB train size), seed 0
- Test: the WikiText-103 test split (2,183 paragraphs)
