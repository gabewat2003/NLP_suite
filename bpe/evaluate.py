"""Tokenizer metrics shared by the eval and comparison scripts."""
import re

import config


def load_terms(path=None):
    path = path or (config.ROOT / "eval" / "probe_terms.txt")
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


def encode_all(tok, texts):
    return [tok.encode(t) for t in texts]


def fertility(texts, encoded):
    """Tokens per whitespace-separated word (lower is better)."""
    words = sum(len(t.split()) for t in texts)
    return sum(map(len, encoded)) / max(words, 1)


def bytes_per_token(texts, encoded):
    """UTF-8 bytes per token (higher is better)."""
    return sum(len(t.encode("utf-8")) for t in texts) / max(sum(map(len, encoded)), 1)


def term_pieces(tok, term):
    """Pieces for a term as it appears mid-sentence (with its leading space)."""
    return tok.pieces(" " + term)


def term_fertility(tok, terms):
    return sum(len(tok.encode(" " + t)) for t in terms) / len(terms)


def unk_rate(tok, encoded):
    unk = tok.special_ids["<unk>"]
    return sum(ids.count(unk) for ids in encoded) / max(sum(map(len, encoded)), 1)


def redaction_integrity(tok, texts, encoded):
    """Share of redaction-token occurrences that come out as exactly their one id."""
    want = got = 0
    for t, ids in zip(texts, encoded):
        for s in config.REDACTION_TOKENS:
            n = len(re.findall(re.escape(s), t))
            want += n
            got += min(n, ids.count(tok.special_ids[s]))
    single = all(tok.encode(s) == [tok.special_ids[s]] for s in config.REDACTION_TOKENS)
    return (got / want if want else 1.0) if single else 0.0


def all_metrics(tok, texts, terms):
    enc = encode_all(tok, texts)
    return {
        "fertility": fertility(texts, enc),
        "bytes_per_token": bytes_per_token(texts, enc),
        "term_fertility": term_fertility(tok, terms),
        "unk_rate": unk_rate(tok, enc),
        "redaction_integrity": redaction_integrity(tok, texts, enc),
    }
