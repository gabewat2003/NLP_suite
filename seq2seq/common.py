"""Helpers shared by the seq2seq run scripts."""
import json

import torch

import config
from eval import plotstyle

RUNS = config.ROOT / "seq2seq" / "runs"


def save_run(name, model, extra):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    (d / "run.json").write_text(json.dumps({"hparams": model.hparams, **extra}, indent=2, default=str))
    return d


def attention_heatmap(attn, src_labels, tgt_labels, title, fname, max_src=None):
    """attn: (T, S) array."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plotstyle.apply()
    cmap = LinearSegmentedColormap.from_list("seq", [plotstyle.SURFACE, plotstyle.SERIES[0]])
    if max_src:
        attn, src_labels = attn[:, :max_src], src_labels[:max_src]
    T, S = attn.shape
    fig, ax = plt.subplots(figsize=(max(4, 0.32 * S + 1.5), max(3, 0.32 * T + 1.2)))
    im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(S), src_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(T), tgt_labels, fontsize=8)
    ax.set_xlabel("source token")
    ax.set_ylabel("output token")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, label="attention weight")
    return plotstyle.save(fig, fname)
