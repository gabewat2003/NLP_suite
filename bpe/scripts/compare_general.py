"""The controlled comparison: CFPB-trained vs general-trained BPE at matched
vocabulary size, each evaluated on CFPB test and general test (the 2x2), plus a
vocabulary-size sweep and an uncontrolled GPT-2 reference row.

Both 16k tokenizers are truncated to each smaller size (BPE merges are a
prefix process, so this equals retraining; see test_truncated_equals_retrained).

    python -m bpe.scripts.compare_general
"""
import argparse

import config
from bpe.evaluate import bytes_per_token, encode_all, fertility, load_terms, term_fertility, term_pieces
from bpe.tokenizer import Tokenizer
from corpus.data import load_split
from eval import plotstyle
from eval.metrics_io import write_metrics

SIZES = [1000, 2000, 4000, 8000, 16000]
ART = config.ROOT / "bpe" / "artifacts"


def gpt2_row(cfpb_texts, gen_texts, terms):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    row = {"vocab_size": enc.n_vocab}
    for name, texts in (("cfpb", cfpb_texts), ("general", gen_texts)):
        ids = [enc.encode_ordinary(t) for t in texts]
        row[f"fertility_{name}"] = fertility(texts, ids)
        row[f"bytes_per_token_{name}"] = bytes_per_token(texts, ids)
    row["term_fertility"] = sum(len(enc.encode_ordinary(" " + t)) for t in terms) / len(terms)
    row["pieces"] = {t: [enc.decode([i]).replace(" ", "▁") for i in enc.encode_ordinary(" " + t)] for t in terms}
    return row


def plot(sweep, gpt2):
    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, test in zip(axes, ("cfpb", "general")):
        for i, trained in enumerate(("cfpb", "general")):
            xs = [r["vocab_size"] for r in sweep if r["trained_on"] == trained]
            ys = [r[f"fertility_{test}"] for r in sweep if r["trained_on"] == trained]
            ax.plot(xs, ys, marker="o", color=plotstyle.SERIES[i],
                    label=f"trained on {'CFPB' if trained == 'cfpb' else 'WikiText'}")
        ax.axhline(gpt2[f"fertility_{test}"], color=plotstyle.NEUTRAL, linestyle="--", linewidth=1)
        ax.annotate("GPT-2 (50k vocab, reference)", (SIZES[0], gpt2[f"fertility_{test}"]),
                    xytext=(0, 5), textcoords="offset points", fontsize=8, color=plotstyle.TEXT_2,
                    va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES, [f"{s // 1000}k" for s in SIZES])
        ax.set_xlabel("vocabulary size")
        ax.set_title(f"Evaluated on {'CFPB test' if test == 'cfpb' else 'WikiText test'}")
    axes[0].set_ylabel("tokens per word (lower is better)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    return plotstyle.save(fig, "bpe_fertility_vs_vocab.png")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    full = {"cfpb": Tokenizer.load(ART / "cfpb_16k"), "general": Tokenizer.load(ART / "general_16k")}
    cfpb_texts = list(load_split("test", columns=["narrative"])["narrative"])
    gen_texts = [l for l in open(config.PROCESSED / "general_test.txt", encoding="utf-8").read().splitlines() if l]
    terms = load_terms()
    sweep = []
    for v in SIZES:
        for trained, big in full.items():
            tok = big.truncated(v)
            if v == 8000:
                tok.save(ART / f"{trained}_8k")
            row = {"vocab_size": v, "trained_on": trained, "term_fertility": term_fertility(tok, terms)}
            for name, texts in (("cfpb", cfpb_texts), ("general", gen_texts)):
                enc = encode_all(tok, texts)
                row[f"fertility_{name}"] = fertility(texts, enc)
                row[f"bytes_per_token_{name}"] = bytes_per_token(texts, enc)
            sweep.append(row)
            print({k: round(x, 4) if isinstance(x, float) else x for k, x in row.items()}, flush=True)
    gpt2 = gpt2_row(cfpb_texts, gen_texts, terms)
    t8 = {k: v.truncated(8000) for k, v in full.items()}
    terms_table = [{"term": t, "cfpb": term_pieces(t8["cfpb"], t), "general": term_pieces(t8["general"], t),
                    "gpt2": gpt2["pieces"][t]} for t in terms]
    gpt2.pop("pieces")
    write_metrics("bpe_compare", {"sweep": sweep, "gpt2": gpt2, "terms_8k": terms_table},
                  {"sizes": SIZES, "cfpb_hash": full["cfpb"].meta.get("corpus_hash"),
                   "general_hash": full["general"].meta.get("corpus_hash")})
    print("figure:", plot(sweep, gpt2))


if __name__ == "__main__":
    main()
