"""Where does Task B attention go? Privacy-safe by design: no narrative text is
published. Instead:

1. Across 2,000 test complaints, each source token's share of attention vs its
   share of tokens ("lift"); the top tokens by lift are plotted.
2. Attention mass on redaction tokens and punctuation vs their frequency.
3. For 5 correct and 5 wrong predictions: only the 8 most-attended tokens.

    python -m seq2seq.scripts.plot_attention
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

import config
from eval import plotstyle
from eval.metrics_io import write_metrics
from seq2seq.common import RUNS
from seq2seq.model import Seq2Seq
from seq2seq.train import pad_batch

N = 2000


def load():
    from seq2seq.data_taskB import TaskB
    run = json.loads((RUNS / "taskB_seed0" / "run.json").read_text())
    cfg = run["cfg"]
    task = TaskB(cfg["tokenizer"], cfg["L"])
    hp = run["hparams"]
    model = Seq2Seq(hp["src_vocab"], hp["tgt_vocab"], emb=hp["emb"], hid=hp["hid"], att=hp["att"],
                    dropout=hp["dropout"], attention=hp["attention"])
    model.load_state_dict(torch.load(RUNS / "taskB_seed0" / "model.pt"))
    model.eval()
    return task, model


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    task, model = load()
    if model.decoder.att is None:
        raise SystemExit("chosen Task B model has no attention")
    preds = json.loads((RUNS / "taskB_test_preds_seed0.json").read_text())
    src, gold = task.data["test"][:N], task.labels["test"][:N]
    att_mass, tok_count = defaultdict(float), defaultdict(int)
    per_example = []
    for i in range(0, N, 128):
        x, xl = pad_batch(src[i:i + 128])
        out, attn = model.greedy(x, xl, 30, return_attn=True)
        for b in range(x.size(0)):
            ids = src[i + b]
            toks = out[b].tolist()
            T = toks.index(2) + 1 if 2 in toks else len(toks)
            a = attn[b, :T, :len(ids)].sum(0).numpy()
            a = a / a.sum()
            for t, w in zip(ids, a):
                att_mass[t] += w
                tok_count[t] += 1
            per_example.append(a)
    total_att = sum(att_mass.values())
    total_tok = sum(tok_count.values())
    lift = {t: (att_mass[t] / total_att) / (tok_count[t] / total_tok) for t in tok_count if tok_count[t] >= 50}
    top = sorted(lift, key=lambda t: -lift[t])[:15]
    bottom = sorted(lift, key=lambda t: lift[t])[:15]
    sp = {s: task.tok.special_ids[s] for s in config.REDACTION_TOKENS}
    groups = {name: [tid] for name, tid in sp.items()}
    groups["punctuation"] = [t for t in tok_count if not any(ch.isalnum() for ch in task.tok.token_text(t))
                             and t not in sp.values()]
    group_stats = {g: {"attention_share": sum(att_mass[t] for t in ids) / total_att,
                       "token_share": sum(tok_count[t] for t in ids) / total_tok} for g, ids in groups.items()}
    correct = [i for i in range(N) if preds[i] == gold[i]][:5]
    wrong = [i for i in range(N) if preds[i] != gold[i]][:5]
    examples = []
    for i in correct + wrong:
        a = per_example[i]
        order = np.argsort(-a)[:8]
        examples.append({"gold": gold[i], "pred": preds[i], "correct": preds[i] == gold[i],
                         "top_tokens": [(task.tok.token_text(src[i][j]), round(float(a[j]), 3)) for j in order]})
    write_metrics("s2s_taskB_attention", {
        "top_lift": [(task.tok.token_text(t), round(lift[t], 2)) for t in top],
        "bottom_lift": [(task.tok.token_text(t), round(lift[t], 2)) for t in bottom],
        "groups": group_stats, "examples": examples}, {"n_test": N})

    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    labels = [task.tok.token_text(t).replace("▁", " ").strip() or "(space)" for t in top][::-1]
    vals = [lift[t] for t in top][::-1]
    ax.barh(range(len(vals)), vals, color=plotstyle.SERIES[0], height=0.6)
    ax.set_yticks(range(len(vals)), labels, fontsize=9)
    ax.axvline(1.0, color=plotstyle.NEUTRAL, linewidth=1, linestyle="--")
    ax.text(1.0, len(vals) - 0.4, " uniform", fontsize=8, color=plotstyle.TEXT_2)
    ax.set_xlabel("attention share ÷ token share (tokens seen ≥ 50 times)")
    ax.set_title("Task B: tokens the decoder attends to most")
    ax.grid(axis="y", visible=False)
    print("figure:", plotstyle.save(fig, "s2s_taskB_attention_lift.png"))
    print("top:", [(task.tok.token_text(t), round(lift[t], 1)) for t in top])
    print("groups:", group_stats)


if __name__ == "__main__":
    main()
