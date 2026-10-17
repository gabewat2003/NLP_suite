import json
from collections import Counter

def get_pair_counts(word_freqs):
    """Count frequencies of adjacent symbol pairs across all words in the corpus."""
    pairs = Counter()
    for word, freq in word_freqs.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(pair, word_freqs):
    """Merge all occurrences of a given symbol pair in the vocabulary word frequencies."""
    merged = {}
    for word, freq in word_freqs.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                out.append(word[i] + word[i+1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        merged[tuple(out)] = freq
    return merged

def train(word_freqs, num_merges):
    """Train BPE tokenizer on word frequencies for a given number of merges."""
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        word_freqs = merge_pair(best, word_freqs)
        merges.append(best)
    return merges

class Tokenizer:
    def __init__(self, merges=None):
        self.merges = merges or []
        self.vocab = {}
        self.inverse_vocab = {}
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary mapping from tokens to integer IDs based on learned merges."""
        # Start with a base set of unique characters/symbols derived from merges
        vocab_set = set()
        for a, b in self.merges:
            vocab_set.add(a)
            vocab_set.add(b)
            vocab_set.add(a + b)
        
        # Sort for deterministic ID assignment
        sorted_tokens = sorted(list(vocab_set))
        
        # Assign IDs (starting from 0)
        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    @classmethod
    def from_corpus(cls, corpus_text, num_merges):
        """Train tokenizer directly from raw corpus text."""
        # Preprocess corpus into word frequencies with '_' end-of-word marker
        words = corpus_text.strip().split()
        word_freqs = Counter()
        for word in words:
            # Represent word as tuple of characters ending with '_'
            word_tuple = tuple(list(word) + ["_"])
            word_freqs[word_tuple] += 1
            
        merges = train(word_freqs, num_merges)
        return cls(merges=merges)

    def _tokenize_word(self, word):
        """Apply learned merges sequentially to a single word."""
        symbols = list(word) + ["_"]
        
        # Iteratively apply merges in the exact order they were learned
        for merge in self.merges:
            a, b = merge
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text):
        """Encode input text into a list of integer token IDs."""
        words = text.strip().split()
        ids = []
        for word in words:
            symbols = self._tokenize_word(word)
            for sym in symbols:
                if sym in self.vocab:
                    ids.append(self.vocab[sym])
                else:
                    # Fallback for unseen symbols: break down into individual characters
                    for char in sym:
                        if char in self.vocab:
                            ids.append(self.vocab[char])
        return ids

    def decode(self, ids):
        """Decode a list of token IDs back into text string."""
        tokens = [self.inverse_vocab[i] for i in ids if i in self.inverse_vocab]
        concatenated = "".join(tokens)
        # Replace end-of-word markers '_' with spaces and clean up extra spacing
        text = concatenated.replace("_", " ").strip()
        return text

    def save(self, filepath):
        """Save merges to a JSON file."""
        # Convert tuples in merges to lists for JSON serialization
        data = {"merges": [[a, b] for a, b in self.merges]}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load tokenizer merges from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(pair) for pair in data["merges"]]
        return cls(merges=merges)
