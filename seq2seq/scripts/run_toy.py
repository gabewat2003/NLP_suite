"""Correctness gates before real data:

1. copy a random digit string        -> 100% exact match
2. reverse a random digit string     -> >= 99% exact match, attention = anti-diagonal
3. overfit 32 Task A examples        -> train loss ~0, 100% exact match

    python -m seq2seq.scripts.run_toy
"""
import argparse
import random

import numpy as np
import torch

import config
from eval.metrics_io import write_metrics
from seq2seq.common import attention_heatmap, save_run
from seq2seq.data_taskA import make_splits
from seq2seq.model import Seq2Seq
from seq2seq.train import greedy_decode, pad_batch, train
from seq2seq.vocab import SymbolVocab

DIGITS = list("0123456789")


def digit_task(kind, n, rng, lo=3, hi=12):
    data = []
    for _ in range(n):
        s = [rng.choice(DIGITS) for _ in range(rng.randint(lo, hi))]
        data.append((s, s if kind == "copy" else s[::-1]))
    return data


def run_digits(kind, seed):
    rng = random.Random(seed)
    vocab = SymbolVocab(DIGITS)
    enc = lambda d: [(vocab.encode(s), vocab.encode(t)) for s, t in d]
    tr, va, te = (enc(digit_task(kind, n, rng)) for n in (20000, 1000, 1000))
    model = Seq2Seq(len(vocab), len(vocab), emb=32, hid=128, att=128, dropout=0.1)
    model, hist = train(model, tr, va, epochs=8, batch_size=64, seed=seed, max_len=14, patience=3)
    preds = greedy_decode(model, [s for s, _ in te], 14)
    em = float(np.mean([p == t for p, (_, t) in zip(preds, te)]))
    print(f"{kind}: test exact match {em:.4f}")
    if kind == "reverse":
        model.cpu().eval()  # the probe batch is built on CPU; training may have used a GPU
        ex = [vocab.encode(list("3141592653"))]
        src, src_len = pad_batch(ex)
        out, attn = model.greedy(src, src_len, 11, return_attn=True)
        a = attn[0, :10].numpy()
        anti = float(np.mean([a[i, 9 - i] for i in range(10)]))
        attention_heatmap(a, list("3141592653"), [vocab.itos[i] for i in out[0, :10].tolist()],
                          "Reverse task: attention is the anti-diagonal", "s2s_reverse_attention.png")
        save_run(f"toy_{kind}", model, {"history": hist, "test_em": em})
        return em, hist, anti
    save_run(f"toy_{kind}", model, {"history": hist, "test_em": em})
    return em, hist, None


def overfit32(seed):
    sp = make_splits(seed=seed, per_digit_train=200, per_digit_eval=10, per_digit_extrap=1)
    rng = random.Random(seed)
    ex = rng.sample(sp["train"], 32)
    sv, tv = SymbolVocab.build([s for s, _, _ in ex]), SymbolVocab.build([t for _, t, _ in ex])
    data = [(sv.encode(s), tv.encode(t)) for s, t, _ in ex]
    model = Seq2Seq(len(sv), len(tv), emb=64, hid=128, att=128, dropout=0.0)
    model, hist = train(model, data, data, epochs=300, batch_size=32, seed=seed, max_len=16, lr=3e-3,
                        log=lambda *_: None)
    preds = greedy_decode(model, [s for s, _ in data], 16)
    em = float(np.mean([p == t for p, (_, t) in zip(preds, data)]))
    print(f"overfit-32: final train loss {hist[-1]['train_loss']:.5f}, exact match {em:.4f}")
    return em, hist[-1]["train_loss"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    torch.set_num_threads(3)
    copy_em, _, _ = run_digits("copy", args.seed)
    rev_em, _, anti = run_digits("reverse", args.seed)
    of_em, of_loss = overfit32(args.seed)
    res = {"copy_test_em": copy_em, "reverse_test_em": rev_em, "reverse_antidiagonal_mean_weight": anti,
           "overfit32_em": of_em, "overfit32_final_loss": of_loss,
           "gates_passed": copy_em == 1.0 and rev_em >= 0.99 and of_em == 1.0}
    write_metrics("s2s_toy", res, {"seed": args.seed})
    print(res)


if __name__ == "__main__":
    main()
