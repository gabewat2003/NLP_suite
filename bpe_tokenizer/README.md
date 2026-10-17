# BPE Tokenizer

Byte-pair encoding learns a vocabulary of **subword units** from a corpus. Frequent words stay whole, rare words split into reusable pieces, so the model never meets a truly unknown word.

## How it works

Training starts from characters and repeatedly merges the most frequent adjacent pair into a new symbol.

**Worked example.** Corpus word counts (`_` marks end of word):

```
low_ : 5    lower_ : 2    newest_ : 6    widest_ : 3
```

1. Start: every word is a sequence of characters: `l o w _`, `l o w e r _`, `n e w e s t _`, `w i d e s t _`.
2. Count adjacent pairs. `e s` appears 9 times (newest 6 + widest 3), `s t` 9 times, `l o` 7 times...
3. Merge a top pair, `e s` into `es`. Now `newest_` is `n e w es t _`.
4. Recount. `es t` is now the most frequent, so merge into `est`.
5. Repeat for N merges. The ordered list of merges **is** the trained tokenizer.

**Encoding** a new word: split into characters, then apply the learned merges in the order they were learned. `lowest_` becomes `l o w e s t _`, then `lo w est _`, then `low est _`.

**Decoding**: concatenate tokens and turn `_` back into a space.

## Implementation guide

```python
from collections import Counter

def get_pair_counts(word_freqs):          # word_freqs: {("l","o","w","_"): 5, ...}
    pairs = Counter()
    for word, freq in word_freqs.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(pair, word_freqs):
    merged = {}
    for word, freq in word_freqs.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                out.append(word[i] + word[i+1]); i += 2
            else:
                out.append(word[i]); i += 1
        merged[tuple(out)] = freq
    return merged

def train(word_freqs, num_merges):
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs: break
        best = pairs.most_common(1)[0][0]
        word_freqs = merge_pair(best, word_freqs)
        merges.append(best)
    return merges
```

`encode(text)` applies `merges` in order to each word. Map each final token to an integer id via a vocab dict.

## Usage

```bash
python -m bpe_tokenizer.train --corpus data/corpus.txt --merges 1000 --out bpe_merges.json
```

```python
from bpe_tokenizer import Tokenizer
tok = Tokenizer.load("bpe_merges.json")
ids = tok.encode("lowest prices")
assert tok.decode(ids) == "lowest prices"
```

## Tests

- Round trip: `decode(encode(x)) == x` on varied strings
- Known toy corpus produces the expected first merges
- Unseen word still encodes (falls back to characters)

## Things to explore

- How vocabulary size (number of merges) changes average tokens per word
- Compare against a production tokenizer (e.g. `tiktoken`) on the same text
