import pytest
from collections import Counter
from bpe_tokenizer.tokenizer import Tokenizer, get_pair_counts, merge_pair, train

def test_toy_corpus_merges():
    # Toy corpus word counts from README example
    word_freqs = {
        ("l", "o", "w", "_"): 5,
        ("l", "o", "w", "e", "r", "_"): 2,
        ("n", "e", "w", "e", "s", "t", "_"): 6,
        ("w", "i", "d", "e", "s", "t", "_"): 3
    }
    merges = train(word_freqs, num_merges=2)
    # Expected first merges: ('e', 's') followed by ('es', 't') or similar top pairs
    assert len(merges) == 2
    assert ("e", "s") in merges

def test_round_trip():
    corpus = "low lower newest widest lowest prices new wide test cases"
    tokenizer = Tokenizer.from_corpus(corpus, num_merges=10)
    
    test_strings = [
        "lowest prices",
        "newest test",
        "wide"
    ]
    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        assert decoded == s

def test_unseen_word_fallback():
    # Train on a small corpus
    corpus = "apple banana"
    tokenizer = Tokenizer.from_corpus(corpus, num_merges=5)
    
    # Encode an entirely unseen word containing out-of-vocabulary characters/combinations
    unseen = "xylophone"
    ids = tokenizer.encode(unseen)
    decoded = tokenizer.decode(ids)
    
    # Should fall back gracefully to character-level representation without crashing
    assert len(ids) > 0
    assert decoded == unseen
