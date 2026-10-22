# Runtime

Wall-clock times measured during the run that produced `reports/summary.md`.

**Hardware:** Apple M1 Pro laptop (10 CPU cores, 16 GB RAM, integrated GPU via PyTorch MPS), macOS, Python 3.12. Everything except seq2seq Task B training runs on CPU. Several stages ran concurrently and competed for the CPU, so single-stage times on an idle machine will be somewhat shorter.

| Make target | Step | Wall clock | Notes |
|---|---|---:|---|
| `data` | `corpus.download` | network-bound | ~925 MB of parquet from the Hugging Face mirror, checksummed |
| | `corpus.split` | 1.5 min | normalize + dedup 958k narratives |
| | `corpus.general` | ~1 min | WikiText-103 sample (cached after first download) |
| `tokenizer` | `train_bpe` CFPB / WikiText (16k) | 28 s / 51 s | incremental trainer |
| | `compare_general` | 40 s | 2x2, sweep, GPT-2 reference |
| | `bench` | 12.5 min | dominated by the naive trainer's 300 s budget per size |
| `baselines` | `eval.tfidf_baseline` | ~1.5-2 h | Product \| Issue (56 classes) alone took 65 min; the C grid is warm-started |
| `embeddings` | `word2vec.scripts.ablate` | 49 min | 39 runs, 6 workers |
| | `train_w2v` | 1 h 44 min | 3 seeds + BPE-unit model in parallel, 5 epochs each, d = 300 |
| | `crosscheck_gensim` | 8 min | 3 gensim seeds (multi-threaded C) |
| | `neighbors`, `eval_classify` | 3 min | |
| `seq2seq` | `run_toy` | 6.5 min | CPU |
| | `run_taskA` | 30 min | 6 runs (3 seeds × attention on/off), 3 at a time, CPU |
| | `run_taskB --ablate` | ~1 h | 9 runs, 3 sharing the GPU (run in two parts after an interruption) |
| | `run_taskB` (training) | 2.5-3.7 h | 3 seeds sharing the GPU, 4 epochs, L = 256; the upper end includes a pause while the laptop slept |
| | `run_taskB --decode-only` | 20 min | greedy + beam + trie-constrained beam on 20k test complaints × 3 seeds, CPU |
| | `plot_attention` | 15 s | |
| `eval` | `eval.report` | < 1 s | |
| `test` | `pytest` | ~15 s | 64 tests on the synthetic fixture; no download needed |

End to end, the full pipeline takes roughly 8-9 hours on this laptop, more than half of it seq2seq Task B.
