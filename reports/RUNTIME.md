# Runtime

Wall-clock times from two machines. The first run (M1 laptop, below) produced the BPE benchmark and the word2vec results; the second run (desktop GPU, further down) reproduced the data, tokenizers, toy gates and Task A, and produced the TF-IDF baseline and all Task B results in `reports/summary.md`.

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

## Second run (desktop GPU)

**Hardware:** Intel i7-11700F (8 cores / 16 threads), 16 GB RAM, NVIDIA RTX 3060 12 GB, Windows 11, Python 3.13, PyTorch 2.14 + CUDA 13. Task B and the toy/Task A runs used the GPU; everything else ran on CPU.

| Make target | Step | Wall clock | Notes |
|---|---|---:|---|
| `data` | download + split + general | ~10 min | |
| `tokenizer` | `train_bpe`, `compare_general` | ~2 min | merges byte-identical to the first run; `bench` not rerun |
| `baselines` | `eval.tfidf_baseline` | 2 h | alongside one word2vec model |
| `embeddings` | `train_w2v`, BPE-unit model | 3 h 28 min | alone except for TF-IDF; ~45 min per epoch. Word-level models run ~60 min per epoch each, ~3x slower than on the M1 |
| `seq2seq` | `run_toy` | 6.5 min | GPU |
| | `run_taskA` | 13 min | 6 runs, 3 at a time, GPU |
| | `run_taskB --ablate` | 1 h 52 min + 23 min | 27 runs (base + 8 knobs, 3 seeds each, 4 epochs, 50k subset), 2-3 workers on the GPU; run in two passes |
| | `run_taskB`, seeds 0-2 | 3 h | 1 process, sharing the GPU with the classification head for the first 2 h; 4-6 epochs each (early stopping), includes test decoding |
| | `run_taskB_cls`, 5 seeds | 1 h 54 min | 1 process; 3-6 epochs each |
| | `run_taskB`, seeds 3-4 | 1 h 28 min | 1 process, GPU to itself (~5 min per epoch); 4 and 6 epochs, includes test decoding |
| | `run_taskB --decode-only`, 5 seeds | 1 h 8 min | greedy + beam + trie-constrained beam on 20k test complaints, 2 workers on CPU (~19 min per seed alone) |
| `eval` | `plot_attention`, `eval.report` | < 1 min | |

**Concurrency limits on this machine.**
- Each CUDA process reserves 5-8 GB of virtual memory. Several of them plus several CPU jobs hit the Windows commit limit (RAM + pagefile) and crashed with out-of-memory errors.
- Five training processes sharing the one GPU ran at a small fraction of single-process speed. Run at most one seq2seq and one classification-head process at a time.
