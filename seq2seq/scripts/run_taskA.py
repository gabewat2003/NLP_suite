"""Task A: spelled-out amounts -> digits. Trains on 1-4 digit dollar amounts and
tests in-range (1-4) and on longer numbers (5-7 digits, extrapolation).
3 seeds, with and without attention.

    python -m seq2seq.scripts.run_taskA
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import config
from eval import plotstyle
from eval.metrics_io import summarize, write_metrics
from seq2seq.common import save_run
from seq2seq.data_taskA import EXTRAP_DIGITS, TRAIN_DIGITS, make_splits
from seq2seq.metrics import by_bucket, char_accuracy, exact_match
from seq2seq.model import Seq2Seq
from seq2seq.train import greedy_decode, train
from seq2seq.vocab import UNK, SymbolVocab

EPOCHS = 12


def one(args):
    seed, attention = args
    torch.set_num_threads(2)
    sp = make_splits(seed=config.SEED)  # data fixed; the model seed varies
    sv = SymbolVocab.build([s for s, _, _ in sp["train"]])
    tv = SymbolVocab.build([t for _, t, _ in sp["train"]])
    enc = {k: [(sv.encode(s), tv.encode(t)) for s, t, _ in v] for k, v in sp.items()}
    model = Seq2Seq(len(sv), len(tv), emb=128, hid=256, att=256, dropout=0.2, attention=attention)
    tag = f"taskA_{'att' if attention else 'noatt'}_seed{seed}"
    model, hist = train(model, enc["train"], enc["val"], epochs=EPOCHS, batch_size=64, seed=seed, max_len=16,
                        log=lambda m: print(f"[{tag}] {m}", flush=True))
    out = {"history": hist}
    for split in ("test_in", "test_out"):
        preds = [tv.decode(p) for p in greedy_decode(model, [s for s, _ in enc[split]], 16)]
        golds = [t for _, t, _ in sp[split]]
        out[split] = {"em": exact_match(preds, golds), "char_acc": char_accuracy(preds, golds),
                      "by_digits": by_bucket(preds, golds, [d for _, _, d in sp[split]])}
        if split == "test_out":
            out["failures"] = [{"src": " ".join(s), "gold": "".join(g), "pred": "".join(p), "digits": d,
                                "src_has_unseen_word": any(sv.stoi.get(w, UNK) == UNK for w in s)}
                               for (s, g, d), p in zip(sp[split], preds) if p != g]
    save_run(tag, model, {"seed": seed, "attention": attention, "test_in_em": out["test_in"]["em"]})
    return seed, attention, out


def plot(curves):
    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    digits = list(TRAIN_DIGITS) + list(EXTRAP_DIGITS)
    ax.axvspan(4.5, 7.5, color=plotstyle.GRID, alpha=0.5, linewidth=0)
    ax.text(4.6, 1.02, "longer than any training amount", fontsize=8, color=plotstyle.TEXT_2, va="bottom")
    for i, (label, (mean, sd)) in enumerate(curves.items()):
        m, s = np.array(mean), np.array(sd)
        style = dict(linestyle="-", marker="o") if i == 0 else dict(linestyle="--", marker="o", markerfacecolor=plotstyle.SURFACE,
                                                                     markersize=10, linewidth=1.5)
        ax.plot(digits, m, color=plotstyle.SERIES[i], label=label, zorder=3 - i, **style)
        ax.fill_between(digits, m - s, m + s, color=plotstyle.SERIES[i], alpha=0.15, linewidth=0)
    ax.set_xticks(digits)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("digits in the dollar amount")
    ax.set_ylabel("exact match (mean ± sd, 3 seeds)")
    ax.set_title("Task A: amount normalization, in range vs extrapolation")
    ax.legend(loc="lower left", title="the two curves coincide", title_fontsize=8)
    return plotstyle.save(fig, "s2s_taskA_extrapolation.png")


def replot():
    """Redraw the figure from reports/metrics/s2s_taskA.json without retraining."""
    from eval.metrics_io import read_metrics
    r = read_metrics("s2s_taskA")["metrics"]
    curves = {}
    for key, label in (("attention", "with attention"), ("no_attention", "no attention")):
        e = r[key]["em_by_digits"]
        curves[label] = ([v["mean"] for v in e.values()], [v["sd"] for v in e.values()])
    return plot(curves)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = [(s, a) for a in (True, False) for s in config.SEEDS]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(one, jobs))
    res, curves = {}, {}
    for attention in (True, False):
        runs = [o for s, a, o in results if a == attention]
        key = "attention" if attention else "no_attention"
        digits = list(TRAIN_DIGITS) + list(EXTRAP_DIGITS)
        per_digit = [[{**r["test_in"]["by_digits"], **r["test_out"]["by_digits"]}[d] for d in digits] for r in runs]
        arr = np.array(per_digit)
        res[key] = {"test_in_em": summarize([r["test_in"]["em"] for r in runs]),
                    "test_out_em": summarize([r["test_out"]["em"] for r in runs]),
                    "test_in_char_acc": summarize([r["test_in"]["char_acc"] for r in runs]),
                    "em_by_digits": {d: summarize(arr[:, i]) for i, d in enumerate(digits)}}
        curves["with attention" if attention else "no attention"] = (arr.mean(0), arr.std(0, ddof=1))
        if attention:
            res["failures_seed0"] = next(o for s, a, o in results if a and s == config.SEEDS[0])["failures"]
    write_metrics("s2s_taskA", res, {"epochs": EPOCHS, "train_digits": TRAIN_DIGITS,
                                      "extrap_digits": EXTRAP_DIGITS, "seeds": list(config.SEEDS)})
    print("figure:", plot(curves))
    for k in ("attention", "no_attention"):
        print(k, {d: round(v["mean"], 3) for d, v in res[k]["em_by_digits"].items()})


if __name__ == "__main__":
    main()
