"""Task B: complaint narrative -> "Product | Issue" label string.

Source: the first L BPE tokens of the normalized narrative, cut back so it never
ends mid-word. Target: the label string in the same BPE tokenizer, re-indexed
into a small target vocabulary (only tokens that occur in labels). Rare labels
collapse to "Other | Other" (see eval/labels.py), fit on train only.
"""
import pickle

import config
from bpe.tokenizer import Tokenizer
from corpus.data import load_split
from eval.labels import label_space, labels
from seq2seq.vocab import SymbolVocab

TOKENIZERS = {"cfpb": config.ROOT / "bpe" / "artifacts" / "cfpb_8k",
              "general": config.ROOT / "bpe" / "artifacts" / "general_8k"}


def truncate_word_safe(ids, L, tok):
    """First <= L ids, dropping a trailing partial word (a cut never splits a word)."""
    if len(ids) <= L:
        return ids
    cut = L
    while cut > 0 and not _starts_word(tok, ids[cut]):
        cut -= 1
    return ids[:cut] if cut > 0 else ids[:L]


def _starts_word(tok, i):
    t = tok.token_text(i)
    return t.startswith("▁") or i in tok._special_id_set or not t[:1].isalnum()


def _cache_path(tokenizer, split, max_tokens):
    return config.PROCESSED / f"taskB_{tokenizer}_{split}_{max_tokens}.pkl"


def encoded_split(tokenizer, split, max_tokens=256):
    """Token ids for each narrative (first max_tokens, word-safe), cached."""
    p = _cache_path(tokenizer, split, max_tokens)
    if p.exists():
        return pickle.loads(p.read_bytes())
    tok = Tokenizer.load(TOKENIZERS[tokenizer])
    df = load_split(split, columns=["complaint_id", "narrative", "product", "issue"])
    ids = []
    for t in df["narrative"]:
        # encode a generous character prefix only; avoids encoding 5k-word complaints in full
        ids.append(truncate_word_safe(tok.encode(t[: max_tokens * 12]), max_tokens, tok))
    out = {"ids": ids, "df": df[["complaint_id", "product", "issue"]].reset_index(drop=True)}
    p.write_bytes(pickle.dumps(out))
    return out


class TaskB:
    def __init__(self, tokenizer="cfpb", L=128, limit_train=None):
        self.tok = Tokenizer.load(TOKENIZERS[tokenizer])
        self.L = L
        tr = encoded_split(tokenizer, "train")
        self.keep = label_space(tr["df"])
        self.data, self.labels = {}, {}
        for split in ("train", "val", "test"):
            enc = tr if split == "train" else encoded_split(tokenizer, split)
            df = enc["df"]
            lab = list(labels(df, "product_issue", self.keep))
            src = [truncate_word_safe(i, L, self.tok) for i in enc["ids"]]
            if split == "train" and limit_train:
                src, lab = src[:limit_train], lab[:limit_train]
            self.labels[split] = lab
            self.data[split] = src
        label_set = sorted(set(self.labels["train"]))
        self.label_pieces = {l: [str(i) for i in self.tok.encode(" " + l)] for l in label_set}
        self.tgt_vocab = SymbolVocab.build(self.label_pieces.values())
        self.label_ids = {l: self.tgt_vocab.encode(p) for l, p in self.label_pieces.items()}
        self.id_to_label = {tuple(v): k for k, v in self.label_ids.items()}
        self.src_vocab_size = len(self.tok)

    def pairs(self, split):
        return [(s, self.label_ids[l]) for s, l in zip(self.data[split], self.labels[split])]

    def to_label(self, ids):
        """Predicted id sequence -> label string, or None if it isn't a valid label."""
        return self.id_to_label.get(tuple(ids))
