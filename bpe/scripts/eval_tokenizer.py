"""Evaluate one tokenizer on a split: fertility, compression, finance-term
fertility, <unk> rate, redaction integrity. Also writes the per-term table.

    python -m bpe.scripts.eval_tokenizer --tokenizer bpe/artifacts/cfpb_8k --split test
"""
import argparse
from pathlib import Path

import config
from bpe.evaluate import all_metrics, load_terms, term_pieces
from bpe.tokenizer import Tokenizer
from corpus.data import load_split
from eval.metrics_io import write_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--name", default="bpe_eval")
    args = ap.parse_args()
    tok = Tokenizer.load(args.tokenizer)
    texts = list(load_split(args.split, columns=["narrative"])["narrative"])
    terms = load_terms()
    m = all_metrics(tok, texts, terms)
    m["vocab_size"] = len(tok)
    write_metrics(args.name, m, {"tokenizer": args.tokenizer, "split": args.split,
                                 "corpus_hash": tok.meta.get("corpus_hash")})
    for k, v in m.items():
        print(f"{k:22s} {v:.4f}" if isinstance(v, float) else f"{k:22s} {v}")


if __name__ == "__main__":
    main()
