"""Pre-tokenization and the byte <-> printable-unicode tables (GPT-2 style).

BPE never merges across the boundaries drawn here. Special tokens are split out
before the regex runs so they are never broken apart.
"""
import re as std_re
from collections import Counter

import regex

PATTERN_VERSION = "gpt2-v1"
PRETOK = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode():
    """Map each of the 256 byte values to a printable unicode character.

    Printable latin-1 bytes map to themselves; the rest are shifted to 256+.
    This lets merges be stored as readable strings ("Ġpayment" = " payment").
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE2UNI = bytes_to_unicode()
UNI2BYTE = {u: b for b, u in BYTE2UNI.items()}
BASE_SYMBOLS = [BYTE2UNI[b] for b in range(256)]


def special_splitter(specials):
    if not specials:
        return None
    # longest first so a special that prefixes another can't win early
    alts = sorted(specials, key=len, reverse=True)
    return std_re.compile("(" + "|".join(std_re.escape(t) for t in alts) + ")")


def split_specials(text, splitter):
    """Yield (is_special, segment) pairs; segments concatenate back to text."""
    if splitter is None:
        if text:
            yield False, text
        return
    for i, part in enumerate(splitter.split(text)):
        if part:
            yield i % 2 == 1, part


def pretokenize(text):
    return PRETOK.findall(text)


def to_symbols(piece):
    return tuple(BYTE2UNI[b] for b in piece.encode("utf-8"))


def symbols_to_bytes(sym_string):
    return bytes(UNI2BYTE[c] for c in sym_string)


def word_freqs(texts, specials=()):
    """Count each pre-token (as a tuple of byte symbols) across texts.

    Training on {pre-token: count} instead of the raw stream is what makes BPE
    training tractable: real text repeats the same pre-tokens constantly.
    """
    splitter = special_splitter(specials)
    raw = Counter()
    for t in texts:
        for is_special, seg in split_specials(t, splitter):
            if not is_special:
                raw.update(PRETOK.findall(seg))
    return Counter({to_symbols(p): c for p, c in raw.items()})
