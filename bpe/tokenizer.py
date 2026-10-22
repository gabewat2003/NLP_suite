"""Byte-level BPE tokenizer: train, encode, decode, save, load.

Vocabulary ids: special tokens first (0..k-1, in config.SPECIALS order), then
the 256 byte symbols in byte order, then one id per merge in rank order.
"""
import json
from pathlib import Path

import config
from bpe.pretokenize import (BASE_SYMBOLS, PATTERN_VERSION, PRETOK, UNI2BYTE,
                             special_splitter, split_specials, to_symbols, word_freqs)
from bpe.train import merge_word, train_incremental, train_naive


class Tokenizer:
    def __init__(self, merges, specials=None, meta=None):
        self.specials = list(config.SPECIALS if specials is None else specials)
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.meta = dict(meta or {})
        self.id_to_token = list(self.specials) + list(BASE_SYMBOLS)
        self.vocab = {t: i for i, t in enumerate(self.id_to_token)}
        for a, b in self.merges:
            t = a + b
            if t not in self.vocab:  # two merge paths can build the same string
                self.vocab[t] = len(self.id_to_token)
                self.id_to_token.append(t)
        self.special_ids = {t: self.vocab[t] for t in self.specials}
        self._special_id_set = set(self.special_ids.values())
        self._splitter = special_splitter(self.specials)
        self._cache = {}

    # ---- training -------------------------------------------------------
    @classmethod
    def train(cls, texts, vocab_size, specials=None, trainer="incremental", meta=None, progress=None):
        specials = list(config.SPECIALS if specials is None else specials)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(specials)}")
        wf = word_freqs(texts, specials)
        if trainer == "naive":
            merges = train_naive(wf, n_merges)
        else:
            merges = train_incremental(wf, n_merges, progress=progress)
        meta = dict(meta or {})
        meta.update(n_merges=len(merges), vocab_size_requested=vocab_size, trainer=trainer,
                    pattern_version=PATTERN_VERSION)
        return cls(merges, specials, meta)

    def truncated(self, vocab_size):
        """The tokenizer this training run would have produced at a smaller vocab.

        Greedy BPE is a prefix process: training to V merges and keeping the
        first k gives exactly what training to k merges gives.
        """
        n = vocab_size - 256 - len(self.specials)
        meta = dict(self.meta, n_merges=min(n, len(self.merges)), vocab_size_requested=vocab_size)
        return Tokenizer(self.merges[:n], self.specials, meta)

    # ---- encoding -------------------------------------------------------
    def __len__(self):
        return len(self.id_to_token)

    def _encode_piece(self, piece):
        ids = self._cache.get(piece)
        if ids is not None:
            return ids
        syms = to_symbols(piece)
        ranks = self.ranks
        while len(syms) > 1:
            best, best_rank = None, None
            for p in zip(syms, syms[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            syms = merge_word(syms, best)
        ids = [self.vocab[s] for s in syms]
        self._cache[piece] = ids
        return ids

    def encode(self, text, bos=False, eos=False):
        out = [self.special_ids["<bos>"]] if bos else []
        for is_special, seg in split_specials(text, self._splitter):
            if is_special:
                out.append(self.special_ids[seg])
            else:
                for piece in PRETOK.findall(seg):
                    out.extend(self._encode_piece(piece))
        if eos:
            out.append(self.special_ids["<eos>"])
        return out

    def encode_batch(self, texts, **kw):
        return [self.encode(t, **kw) for t in texts]

    def clear_cache(self):
        self._cache = {}

    # ---- decoding -------------------------------------------------------
    def decode(self, ids, skip_specials=False):
        out, buf = [], bytearray()
        for i in ids:
            if i in self._special_id_set:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_specials:
                    out.append(self.id_to_token[i])
            else:
                buf.extend(UNI2BYTE[c] for c in self.id_to_token[i])
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def token_text(self, i):
        """Readable form of one token: leading space shown as '▁'."""
        if i in self._special_id_set:
            return self.id_to_token[i]
        s = bytes(UNI2BYTE[c] for c in self.id_to_token[i]).decode("utf-8", errors="replace")
        return s.replace(" ", "▁")

    def pieces(self, text):
        return [self.token_text(i) for i in self.encode(text)]

    # ---- persistence ----------------------------------------------------
    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False, indent=0) + "\n",
                                        encoding="utf-8")
        (out / "merges.txt").write_text("".join(f"{a} {b}\n" for a, b in self.merges), encoding="utf-8")
        cfg = dict(self.meta, specials=self.specials, vocab_size=len(self))
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        return out

    @classmethod
    def load(cls, in_dir):
        d = Path(in_dir)
        cfg = json.loads((d / "config.json").read_text())
        merges = [tuple(line.split(" ")) for line in
                  (d / "merges.txt").read_text(encoding="utf-8").splitlines() if line]
        specials = cfg.pop("specials")
        cfg.pop("vocab_size", None)
        tok = cls(merges, specials, cfg)
        saved = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        if saved != tok.vocab:
            raise ValueError(f"{d}/vocab.json does not match merges.txt")
        return tok
