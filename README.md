# NLP From Scratch, Pointed at Lending: BPE → word2vec → seq2seq on CFPB Complaints

[![tests](https://github.com/gabewat2003/NLP_suite/actions/workflows/test.yml/badge.svg)](https://github.com/gabewat2003/NLP_suite/actions/workflows/test.yml)

Three core NLP components, written by hand and trained on one shared corpus of 150,000 consumer-finance complaints:
- a byte-pair-encoding tokenizer
- skip-gram word2vec with negative sampling
- an attention-based GRU encoder–decoder

Each one is measured against a plain baseline, so the results say what these methods do and don't buy you on financial text.

**Stack:** Python · NumPy · PyTorch (seq2seq only) · `regex` · scikit-learn (baselines only) · pytest · hypothesis · (cross-checks only) gensim, tiktoken

**Findings at a glance** (details and numbers in §5):
- **BPE wins.** A tokenizer trained on complaints needs 19% fewer tokens per word than a general one of the same size, and keeps terms like `forbearance` and `FDCPA` whole.
- **word2vec loses to TF-IDF, usefully.** Averaged word vectors classify Product well above random vectors and on par with gensim, but a bag of n-grams still beats them.
- **seq2seq ties TF-IDF on complaint labels, and the decoder isn't why.** Its gains come from the domain tokenizer and word2vec-initialized embeddings. The same encoder with a plain classification head does slightly better and varies less across seeds.
- **seq2seq learns templates, not rules.** It normalizes dollar amounts perfectly up to 4 digits and gets every 5-7 digit amount wrong.

---

## 1. Why this exists

Almost every modern text model starts with the same three ideas: split text into subword pieces, give those pieces vectors, and let a network map one sequence to another. I built each one from scratch so I can explain:
- what a tokenizer does to a word like `forbearance`
- why `charge` gets one vector for credit-card charges, fees and charge-offs
- where an encoder–decoder runs out of room

I would not train these from scratch in production, because pretrained models are better. The point here is **understanding plus honest measurement**, so each component ends in a number next to a boring baseline.

Why finance: lending teams read a lot of unstructured text, including complaint narratives, dispute notes and collections comments. That text is full of terms a general-purpose tokenizer shreds (`FDCPA`, `reinvestigation`, `garnishment`) and of redaction artifacts (`XXXX`, `{$1,200.00}`). It's also the unstructured side of the same business my credit-risk and fraud projects model from tabular data. A complaint says in words what a delinquency flag says in a column.

---

## 2. The corpus (shared by all three components)

**Source:** the CFPB Consumer Complaint Database (public, US government, CC0). Only complaints where the consumer opted in to publishing the narrative have text.

**The CFPB removed every narrative from its public database in September 2026** (Consumer Complaint Database Release 24), and the official export now has no text column. This project uses a CC0 mirror captured before the removal: Hugging Face `BEE-spoke-data/consumer-finance-complaints`, "has-text" subset, pinned to one revision, with every file checked against its published SHA-256. Details are in [`data/SOURCES.md`](data/SOURCES.md).

| Field used | Role in this project |
|---|---|
| `Consumer complaint narrative` | The text. Training corpus for all three components |
| `Product`, `Issue` | Labels for the downstream evaluations (classification, seq2seq target) |
| `Date received` | Time-based split; also used to check label-taxonomy drift |
| `Complaint ID` | Join key, dedup key, and leakage fingerprint |
| `Company` | **Not** used as a feature |

1.69M narratives in the mirror (March 2015 to February 2024) → 958k in the 2018–2022 window → 681k after dedup → 150k train / 20k val / 20k test.

### 2.1 Things in this data that bite

| Trap | What it looks like | What this project does |
|---|---|---|
| Redaction tokens | `XXXX`, `XX/XX/XXXX`, `{$1,234.56}`, even `{$XXXX}` | Normalized to `<REDACTED>`, `<DATE>`, `<AMT>` before any training (`corpus/normalize.py`). They are registered as special tokens in the tokenizer. In train: 776k `<REDACTED>`, 275k `<DATE>`, 147k `<AMT>` |
| Copy-paste duplicates | Template letters submitted by many people | **29% of the window is duplicates**: 191,793 exact copies plus 85,366 near-copies (same first 200 characters), all dropped, keeping the earliest |
| Taxonomy drift | `Product` names renamed and merged in April 2017 and again in 2023 | Restricted to 2018–2022, where every Product label is constant (`python -m corpus.split --labels` prints the table) |
| Class imbalance | Credit reporting is 39% of train and 52% of test | Macro-F1 reported alongside accuracy; majority-class row in every table |
| Residual PII | Redaction is imperfect | No raw narrative appears in this README, the notebooks or the reports. Examples are synthetic or reduced to single tokens |
| Length | p50 128 words, p99 1,076 | Truncated by *token* count for seq2seq, never mid-word |

### 2.2 Splits

Time-based, not random. Complaint language and the label mix drift: debt collection falls from 20% of train to 12% of test. A random split would hide that.

```
train: 2018-01-01 .. 2021-12-31   150,000 complaints (sampled from 482k)   fit everything here
val:   2022-01-01 .. 2022-06-30    20,000 complaints                        model selection
test:  2022-07-01 .. 2022-12-31    20,000 complaints                        touched once, at the end
```

Every fit step records a hash of the complaint IDs it saw: the tokenizer merges, the word2vec vocabulary, the TF-IDF vectorizer and the seq2seq models. `tests/test_no_leakage.py` fails if any of those hashes isn't the train split's.

---

## 3. Repository layout

```
nlp_suite/
├── README.md
├── Makefile                 # make data | tokenizer | baselines | embeddings | seq2seq | eval | test
├── config.py                # seed, split dates, caps, special tokens: one place
├── requirements.txt         # pinned; requirements-dev.txt adds notebook tooling
├── corpus/                  # download (+ checksums), normalize, dedup, time split, WikiText sample
├── data/SOURCES.md          # where the data came from, row counts, the window decision (data/ itself is gitignored)
├── bpe/                     # component 1: tokenizer          -> bpe/README.md
├── word2vec/                # component 2: embeddings         -> word2vec/README.md
├── seq2seq/                 # component 3: encoder-decoder    -> seq2seq/README.md
├── eval/                    # TF-IDF baseline, shared labels/metrics, probe terms, report builder
├── notebooks/               # 00_pipeline.ipynb + build.py (notebooks are built and executed by script)
├── reports/                 # summary.md, figures/, metrics/*.json, notes.md, taskA_failures.md
└── tests/                   # leakage tests, end-to-end smoke test, synthetic fixture corpus
```

Each component is a flat Python package run from the repository root (`python -m bpe.scripts.train_bpe ...`), with its own `scripts/`, `tests/` and README.

---

## 4. How the three pieces connect

```
        CFPB narratives ──► corpus/ (normalize, dedup, split) ──► train / val / test
                                             │
                    ┌────────────────────────┼─────────────────────────┐
                    ▼                        ▼                         ▼
             bpe/ (merges on train)   word2vec/ (word vocab)    eval/ TF-IDF + LR
                    │  ids                   │                  (the number to beat)
                    ├──────────────┐         │ also trained on BPE units
                    ▼              ▼         ▼
          fertility vs general   seq2seq/ ◄── encoder embeddings initialized
          tokenizer (2×2)        (GRU + attention)  from word2vec (ablation)
                                       │
                                       ▼
                         eval/report.py ──► reports/summary.md
```

Each component works on its own. The links are ablations:

1. **BPE → word2vec:** do vectors for BPE units give misspelled words a usable vector, where word-level vectors have none? **No**, with only 1 hit in 41. The domain tokenizer makes the correct terms single tokens, so misspellings share no pieces with them (`word2vec/README.md` §6.5).
2. **word2vec → seq2seq:** encoder embeddings initialized from word2vec vs random. **The biggest Task B ablation win**: +2.7 points of val exact match (`seq2seq/README.md` §6.3).
3. **BPE → seq2seq:** domain-trained vs general BPE at the same vocabulary size. **The domain tokenizer is better by 5.6 points** (`seq2seq/README.md` §6.3).

---

## 5. Results summary

Every number below is copied from [`reports/summary.md`](reports/summary.md), which `make eval` regenerates. ± is the standard deviation over seeds (3; 5 for Task B). Test split: 20,000 complaints from Jul-Dec 2022.

| Component | Headline question | Metric | Mine | Baseline | Verdict |
|---|---|---|---:|---|---|
| BPE (CFPB-trained, V=8k) | Tokens per word on complaints | fertility (lower is better) | **1.206** | 1.484 (WikiText-trained, V=8k); 1.240 (GPT-2, V=50k) | **Win**: 19% fewer tokens than a general tokenizer of the same size; beats GPT-2's 6x larger vocabulary |
| BPE | Tokens per finance term | term fertility, 50 terms | **1.16** | 2.74 (WikiText-trained); 1.58 (GPT-2) | **Win**: `forbearance`, `FDCPA`, `Equifax` are one token each |
| word2vec (SGNS, d=300) | Do mean-pooled vectors classify Product? | macro-F1 | 0.736 ± 0.004 | **0.757** (TF-IDF + LR); 0.607 ± 0.001 (random vectors); 0.732 ± 0.001 (gensim) | **Loses to TF-IDF**, clearly beats random vectors, matches gensim |
| word2vec | Do neighbors make domain sense? | precision@5, 20 probes, blind-scored | 0.93 | 0.97 (gensim) | Close to the reference |
| seq2seq (GRU + attention) | Amount normalization | exact match: 1-4 digits / 5-7 digits | 1.000 / **0.000** | same with no attention | **Fails to extrapolate**: learned the `$X,XXX.XX` template, not place value |
| seq2seq | Narrative → Product \| Issue (56 labels) | accuracy / macro-F1 | 0.508 ± 0.028 / 0.371 ± 0.013 | 0.501 / 0.361 (TF-IDF + LR); 0.188 (majority) | **Tied with TF-IDF**: +0.7 points of accuracy, a quarter of the seed spread |
| seq2seq encoder + classification head | same task | accuracy / macro-F1 | **0.521 ± 0.013** / **0.369 ± 0.014** | same | **Small win**: +2 points over TF-IDF and steadier than the decoder; macro-F1 tied |

**Where the from-scratch components were competitive, and where they weren't.**
- **BPE: a clean win.** Trained on complaints, it compresses complaint text better than a general tokenizer of the same size and keeps finance terms whole. The win is measured with vocabulary size held fixed, so it isn't a size artifact.
- **word2vec: a useful loss.** The vectors carry real signal (+0.13 macro-F1 over random vectors) and behave like gensim's. But averaging static vectors over a long complaint loses to a bag of n-grams, as expected.
- **seq2seq: a tie, carried by the other two components.** Inside its training range it is perfect at amount normalization and completely fails outside it. On complaint labels I expected it to lose to TF-IDF; it ties: 0.508 vs 0.501 accuracy, with five seeds spread from 0.477 to 0.553. Most of what it has comes from the other two components: word2vec-initialized embeddings were the biggest ablation win (+2.7 points), and the domain tokenizer beat the general one by 5.6. Dropping the decoder for a classification head on the same encoder does better (0.521 ± 0.013): generating a label token by token buys nothing on a fixed 56-label task.

![Task B: per-seed test accuracy against TF-IDF](reports/figures/s2s_taskB_models.png)

| BPE: tokens per word vs vocabulary size | seq2seq Task A: in range vs extrapolation |
|---|---|
| ![BPE fertility](reports/figures/bpe_fertility_vs_vocab.png) | ![Task A extrapolation](reports/figures/s2s_taskA_extrapolation.png) |

**Why ~50% on Product | Issue is reasonable:**
- The target is exact match over 56 consumer-chosen labels; on Product alone, TF-IDF reaches 84%.
- The biggest errors swap neighboring credit-reporting issues that one complaint can legitimately fit.
- One issue, "Improper use of your report", quadruples its share between the training years (4.1%) and the test period (17.5%).

---

## 6. Reproduce

Python 3.11+ (developed on 3.12; also run on 3.13 under Windows). Budget about 2.5 GB of disk for data and trained models, plus about 1 GB for the Python environment (PyTorch).

```bash
git clone https://github.com/gabewat2003/NLP_suite.git && cd NLP_suite
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make data          # download (checksummed) -> normalize -> dedup -> split -> WikiText sample
make tokenizer     # train BPE on train split, 2x2 comparison, vocab sweep, benchmark
make baselines     # TF-IDF + logistic regression (needed by the next two)
make embeddings    # word2vec ablations on val, final model x3 seeds, gensim cross-check, probes
make seq2seq       # toy gates, Task A, Task B ablations + final, attention analysis
make eval          # reports/summary.md
make test          # pytest: unit, property, leakage and end-to-end smoke tests
```

The notebooks are rebuilt and executed by script, after the targets above:

```bash
pip install -r requirements-dev.txt
python notebooks/build.py
```

The first run's full pipeline took roughly 8-9 hours on an M1 Pro laptop, with a smaller Task B budget. The second run's larger Task B budget (27 ablation runs, 5 seq2seq and 5 classification-head seeds) took about 8 hours on an RTX 3060. Every target runs from a clean clone on fixed seeds (`config.SEED`, `config.SEEDS`). Stochastic results are mean ± sd over 3 seeds (5 for Task B). Wall-clock times for both machines are in [`reports/RUNTIME.md`](reports/RUNTIME.md). seq2seq training uses a GPU if one is present (CUDA, else Apple MPS, else CPU; override with `NLP_SUITE_DEVICE`); everything else runs on CPU. On Windows, set `PYTHONUTF8=1`. `make test` and CI need no download: they run the whole pipeline on a small synthetic corpus (`tests/fixtures/make_fixture.py`).

---

## 7. Limitations

- **The narratives are a self-selected sample.** Only consumers who opted in to publication have text, and people who file complaints are not representative of all borrowers. Nothing here estimates population behavior.
- **The data is a frozen snapshot of a withdrawn source.** The CFPB no longer publishes narratives, so this corpus can't be refreshed from the official source. Results describe 2018–2022 complaints.
- **Redaction removes information.** Dates, names and amounts are masked, so no model here can use them. Any claim about "understanding amounts" is limited to what survives redaction.
- **Labels are consumer-chosen.** `Product` and `Issue` are picked from a menu by the complainant and are noisy ground truth with overlapping categories.
- **The test period differs from train.** The time split is deliberate, but test-set numbers mix model quality with label drift (credit reporting grows from 39% to 52%).
- **Small compute, small models.** These are from-scratch models on a laptop and one desktop GPU, not what a tuned pretrained model would do.
- **No claim of production readiness.** This is a learning-and-measurement project.

Hit while running it:
- **The test period's label mix differs sharply from training.** That's the time split doing its job, but it caps every model's score.
- **Template letters shape everything:**
  - 29% of the window was duplicates.
  - `▁1681` (the FCRA's section number) is a single BPE token.
  - `fdcpa`'s nearest word vectors are the FDCPA's own section numbers.
- **Pasted credit reports pollute the text:** glued fragments like `statusclosed` became `balance`'s nearest neighbors.
- **Seeds matter more than most design choices for seq2seq.** Task B's five final seeds span 7.6 points of accuracy, wider than the effect of any single ablation knob. Three seeds made it look 1 point ahead of TF-IDF; five put it level.
- **Two machines.** The word2vec models and the BPE trainer benchmark come from the first run on an M1 laptop. Everything else was rerun on a desktop RTX 3060 (BPE merges came out byte-identical), and the TF-IDF baseline and all Task B results come from that rerun (`reports/RUNTIME.md`). GPU kernels are not bit-deterministic, so Task B reruns match to within seed noise, not to the last digit.
- **A decoder bug** (beam search stopping early) cost 20 points of Task B accuracy before a test with a known right answer caught it. It's documented in `seq2seq/README.md` §4.5.

---

## 8. What I'd do next

- **Task A with a few longer training amounts, or a copy mechanism**, to test whether place value is learnable here at all or whether the model only ever copies output shapes it has seen.
- **A pretrained encoder on Task B** (a small BERT-style model, fine-tuned) to measure how much of the remaining gap is model capacity versus label ambiguity and 2022 drift.
- **Merge the three credit-reporting issues that swap most**, or score them as one, to see how much of the ~50% ceiling is the label menu rather than the models.
- **Re-time the word2vec trainer.** It ran about 3x slower per epoch on the desktop CPU than on the M1, which is worth profiling before anyone reruns it.

---

## 9. How this was built

I chose the project, the domain and the questions each component has to answer. The component specs, code, tests, experiment runs and drafts of these READMEs were produced with help from AI assistants. Every number in this README is generated by the code in this repository and copied from `reports/summary.md`. The correctness checks are there so the results don't rest on trusting anyone's code, mine or an assistant's: round-trip and equivalence tests for BPE, the gradient check and gensim cross-check for word2vec, the toy-task gates for seq2seq.

---

## License

Code: [MIT](LICENSE). Data is not redistributed here; `make data` downloads it from its sources. The CFPB complaint narratives (via the Hugging Face mirror) are CC0. WikiText-103, used only for the BPE comparison, is CC BY-SA 3.0.
