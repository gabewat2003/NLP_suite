"""Naive vs incremental trainer: time to V=8000 at 1k / 10k / 100k complaints,
merge lists checked for equality, plus encoding throughput.

The naive trainer gets a time budget per size; when it runs out, its time is
projected from the measured per-merge cost and marked as projected.

    python -m bpe.scripts.bench [--budget 300]
"""
import argparse
import platform
import resource
import time

import config
from bpe.pretokenize import word_freqs
from bpe.tokenizer import Tokenizer
from bpe.train import train_incremental, train_naive
from corpus.data import load_split
from eval.metrics_io import write_metrics

V = 8000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=300.0, help="seconds per naive run")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    args = ap.parse_args()
    texts = list(load_split("train", columns=["narrative"])["narrative"])
    n_merges = V - 256 - len(config.SPECIALS)
    rows = []
    for n in args.sizes:
        wf = word_freqs(texts[:n], config.SPECIALS)
        t0 = time.perf_counter()
        inc = train_incremental(wf, n_merges)
        t_inc = time.perf_counter() - t0
        t0 = time.perf_counter()
        naive = train_naive(wf, n_merges, max_seconds=args.budget)
        t_naive = time.perf_counter() - t0
        assert naive == inc[:len(naive)], f"trainers disagree at size {n}"
        done = len(naive) == len(inc)
        naive_s = t_naive if done else t_naive / max(len(naive), 1) * len(inc)
        row = {"docs": n, "unique_pretokens": len(wf), "merges": len(inc),
               "incremental_s": round(t_inc, 2), "naive_s": round(naive_s, 1),
               "naive_projected": not done, "naive_merges_run": len(naive),
               "speedup": round(naive_s / t_inc, 1)}
        rows.append(row)
        print(row, flush=True)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if platform.system() == "Darwin" else 1 << 10)
    tok = Tokenizer.load(config.ROOT / "bpe" / "artifacts" / "cfpb_8k")
    test = list(load_split("test", columns=["narrative"])["narrative"])[:5000]
    t0 = time.perf_counter()
    for t in test:
        tok.encode(t)
    per_s = len(test) / (time.perf_counter() - t0)
    hw = f"{platform.machine()} {platform.system()} {platform.mac_ver()[0] or platform.release()}, Python {platform.python_version()}"
    write_metrics("bpe_bench", {"rows": rows, "encode_complaints_per_s": round(per_s, 1),
                                "peak_rss_mb": round(peak_mb), "hardware": hw},
                  {"vocab_size": V, "budget_s": args.budget})
    print(f"encode: {per_s:.0f} complaints/s (cold cache, 5k test docs); peak RSS {peak_mb:.0f} MB; {hw}")


if __name__ == "__main__":
    main()
