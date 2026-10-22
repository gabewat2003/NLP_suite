"""Small symbol vocabularies for the seq2seq tasks (ids 0-3 are fixed specials)."""
import json

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class SymbolVocab:
    def __init__(self, symbols):
        self.itos = SPECIALS + [s for s in symbols if s not in SPECIALS]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(cls, sequences):
        seen = sorted({s for seq in sequences for s in seq})
        return cls(seen)

    def __len__(self):
        return len(self.itos)

    def encode(self, symbols, bos=False, eos=False):
        ids = [self.stoi.get(s, UNK) for s in symbols]
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, BOS):
                out.append(self.itos[i])
        return out

    def to_json(self):
        return json.dumps(self.itos[len(SPECIALS):])

    @classmethod
    def from_json(cls, s):
        return cls(json.loads(s))
