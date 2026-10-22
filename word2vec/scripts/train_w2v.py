"""Train the final embeddings on the full train split: the config chosen by
ablate.py (base config if none), 3 seeds in parallel, plus one BPE-unit model
(seed 0) for the subword comparison. Saves artifacts and the loss curve.

    python -m word2vec.scripts.train_w2v
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import config
from eval import plotstyle
from word2vec.experiment import ART, BASE, chosen_config, fit, load_tokens

BPE_DIR = str(config.ROOT / "bpe" / "artifacts" / "cfpb_8k")


def one(args):
    name, cfg, seed = args
    c = dict(BASE, **cfg)
    tok = None
    if c["units"] == "bpe":
        from bpe.tokenizer import Tokenizer
        tok = Tokenizer.load(BPE_DIR)
    df, toks = load_tokens("train", c["lowercase"], c["units"], tok)
    emb, hist = fit(cfg, seed, toks, df["complaint_id"], BPE_DIR if tok else None)
    emb.save(ART / name)
    (ART / name / "loss.json").write_text(json.dumps(hist))
    print(f"  {name}: vocab {len(emb.words):,}, {emb.meta['train_seconds']:.0f}s, final loss {hist[-1][1]:.4f}",
          flush=True)
    return name, hist


def plot(hists):
    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (name, h) in enumerate(sorted(hists.items())):
        xs = [x / 1e6 for x, _ in h]
        ax.plot(xs, [y for _, y in h], color=plotstyle.SERIES[i % 3], linewidth=1.5, label=name)
    ax.set_xlabel("training tokens seen (millions, before subsampling)")
    ax.set_ylabel("mean SGNS loss per pair")
    ax.set_title("word2vec training loss (3 seeds)")
    ax.legend()
    return plotstyle.save(fig, "w2v_loss.png")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = chosen_config()
    print("config changes vs base:", cfg)
    jobs = [(f"seed{s}", cfg, s) for s in config.SEEDS] + [("bpe_seed0", dict(cfg, units="bpe"), 0)]
    with ProcessPoolExecutor(len(jobs)) as ex:
        hists = dict(ex.map(one, jobs))
    print("figure:", plot({k: v for k, v in hists.items() if k.startswith("seed")}))


if __name__ == "__main__":
    main()
