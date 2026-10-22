"""Train a BPE tokenizer.

    python -m bpe.scripts.train_bpe --corpus cfpb --vocab-size 16000 --out bpe/artifacts/cfpb_16k
    python -m bpe.scripts.train_bpe --corpus general --vocab-size 16000 --out bpe/artifacts/general_16k

--corpus cfpb uses the train split only; its complaint-ID hash is stored in
config.json so tests/test_no_leakage.py can check it.
"""
import argparse
import hashlib
import time

import config
from bpe.tokenizer import Tokenizer
from corpus.data import ids_hash, load_split


def load_corpus(name, limit=None):
    if name == "cfpb":
        df = load_split("train", columns=["complaint_id", "narrative"])
        if limit:
            df = df.iloc[:limit]
        return list(df["narrative"]), {"corpus": "cfpb-train", "corpus_hash": ids_hash(df["complaint_id"]),
                                       "n_docs": len(df)}
    path = config.PROCESSED / "general_train.txt" if name == "general" else name
    raw = open(path, encoding="utf-8").read()
    lines = raw.splitlines()[:limit] if limit else raw.splitlines()
    return lines, {"corpus": str(name), "corpus_hash": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                   "n_docs": len(lines)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="cfpb", help="cfpb | general | path to a text file")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="use only the first N documents")
    ap.add_argument("--trainer", choices=["incremental", "naive"], default="incremental")
    args = ap.parse_args()
    texts, meta = load_corpus(args.corpus, args.limit)
    print(f"Training on {meta['n_docs']:,} docs ({args.corpus}), vocab {args.vocab_size:,}")
    t0 = time.perf_counter()
    tok = Tokenizer.train(texts, args.vocab_size, trainer=args.trainer, meta=meta, progress=2000)
    tok.meta["train_seconds"] = round(time.perf_counter() - t0, 1)
    tok.save(args.out)
    print(f"Saved {len(tok):,}-token vocab to {args.out} in {tok.meta['train_seconds']}s")


if __name__ == "__main__":
    main()
