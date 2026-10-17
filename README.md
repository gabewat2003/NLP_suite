# nlp-from-scratch

A small NLP toolkit built component by component: a **byte-pair-encoding tokenizer**, **word2vec embeddings**, and a **sequence-to-sequence model**. Each piece is a standalone module, and together they form one pipeline: raw text goes in, subword tokens come out, tokens become vectors, and vectors feed a model that transforms one sequence into another.

```
raw text ──► BPE tokenizer ──► token ids ──► word2vec embeddings ──► seq2seq model ──► output text
            (bpe_tokenizer)                   (word2vec)              (seq2seq)
```

## Components

| Module | What it does | Docs |
|---|---|---|
| `bpe_tokenizer/` | Learns a subword vocabulary from a corpus and encodes/decodes text | [README](bpe_tokenizer/README.md) |
| `word2vec/` | Skip-gram with negative sampling; learns word vectors | [README](word2vec/README.md) |
| `seq2seq/` | Encoder-decoder recurrent network with teacher forcing | [README](seq2seq/README.md) |

## Results

<!-- Fill in with real numbers after rerunning. Delete rows you don't have. -->

| Component | Metric | Result |
|---|---|---|
| BPE tokenizer | Vocab size / avg tokens per word | TODO |
| word2vec | Nearest-neighbor examples, analogy accuracy | TODO |
| seq2seq | Task, exact-match / BLEU | TODO |

## Quickstart

```bash
git clone https://github.com/<your-username>/nlp-from-scratch
cd nlp-from-scratch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m bpe_tokenizer.train --corpus data/corpus.txt --merges 1000
python -m word2vec.train --corpus data/corpus.txt --dim 100 --epochs 5
python -m seq2seq.train --task reverse
```

## Project layout

```
nlp-from-scratch/
├── README.md
├── requirements.txt
├── data/                  # not committed; scripts download or document the source
├── bpe_tokenizer/         # tokenizer.py, train.py, tests/
├── word2vec/              # model.py, train.py, evaluate.py, tests/
├── seq2seq/               # model.py, data.py, train.py, tests/
└── notebooks/             # one demo notebook per module, outputs saved
```

## Tech

Python, NumPy, PyTorch, pytest.

## Build checklist (delete once finished)

- [ ] Pick one small public-domain corpus (e.g. a Project Gutenberg book) for all three modules
- [ ] BPE: train, encode, decode, round-trip test
- [ ] word2vec: pairs, negative sampling, training loop, nearest neighbors, plot
- [ ] seq2seq: toy task first (reverse a sequence), then a real one
- [ ] Wire the pipeline together in `notebooks/00_pipeline.ipynb`
- [ ] Fill in the results table with real numbers
