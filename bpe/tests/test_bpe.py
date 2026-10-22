import string

import pytest
from hypothesis import given, settings, strategies as st

import config
from bpe import Tokenizer, train_incremental, train_naive
from bpe.pretokenize import word_freqs
from corpus.data import load_split

FIXTURE_STRINGS = [
    "", "a", "Hello, world!", "I paid <AMT> on <DATE> to <REDACTED>.",
    "café naïve résumé", "emoji 💳🏦 test", "tabs\tand\r\nnewlines\n\n", "  leading and trailing  ",
    "under_score and <pad>-like <bos> text", "<REDACTED><REDACTED>", "numbers 1234567 and 3.14",
    "日本語のテキスト", "'s 're 've 'll",
]


@pytest.fixture(scope="module")
def train_texts(processed_dir):
    return list(load_split("train", processed=processed_dir)["narrative"])


@pytest.fixture(scope="module")
def tok(train_texts):
    return Tokenizer.train(train_texts, vocab_size=600)


def test_toy_corpus_first_merges():
    wf = {tuple("low"): 5, tuple("lower"): 2, tuple("newest"): 6, tuple("widest"): 3}
    # (e,s) and (s,t) tie at 9; the tie-break picks the larger pair, (s,t).
    assert train_naive(wf, 2) == [("s", "t"), ("e", "st")]
    assert train_incremental(wf, 2) == [("s", "t"), ("e", "st")]


@pytest.mark.parametrize("s", FIXTURE_STRINGS)
def test_roundtrip_fixture(tok, s):
    assert tok.decode(tok.encode(s)) == s


@settings(max_examples=300, deadline=None)
@given(st.text())
def test_roundtrip_property(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_naive_equals_incremental(train_texts):
    wf = word_freqs(train_texts, config.SPECIALS)
    assert train_naive(wf, 300) == train_incremental(wf, 300)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.text(alphabet=string.ascii_lowercase[:5] + " ", max_size=30), max_size=12),
       st.integers(1, 40))
def test_naive_equals_incremental_property(texts, n):
    wf = word_freqs(texts)
    assert train_naive(wf, n) == train_incremental(wf, n)


def test_deterministic(train_texts, tmp_path):
    a = Tokenizer.train(train_texts, 500).save(tmp_path / "a")
    b = Tokenizer.train(train_texts, 500).save(tmp_path / "b")
    assert (a / "merges.txt").read_bytes() == (b / "merges.txt").read_bytes()


def test_specials_atomic(tok):
    for t in config.SPECIALS:
        ids = tok.encode(f"x {t} y")
        assert tok.special_ids[t] in ids
        assert tok.encode(t) == [tok.special_ids[t]]


def test_no_unk_emitted(tok, processed_dir):
    unk = tok.special_ids["<unk>"]
    for text in load_split("test", processed=processed_dir)["narrative"]:
        assert unk not in tok.encode(text)


def test_vocab_size_formula(tok):
    assert len(tok) == 256 + len(tok.merges) + len(config.SPECIALS)


def test_cache_cold_vs_warm(tok):
    s = "The collector reported a <AMT> balance to Experian."
    warm = tok.encode(s)
    tok.clear_cache()
    assert tok.encode(s) == warm


def test_save_load(tok, tmp_path):
    tok.save(tmp_path)
    tok2 = Tokenizer.load(tmp_path)
    assert tok2.merges == tok.merges and tok2.vocab == tok.vocab
    s = "forbearance and garnishment"
    assert tok2.encode(s) == tok.encode(s)


def test_truncated_equals_retrained(train_texts):
    big = Tokenizer.train(train_texts, 700)
    small = Tokenizer.train(train_texts, 450)
    assert big.truncated(450).merges == small.merges


def test_bos_eos(tok):
    ids = tok.encode("hi", bos=True, eos=True)
    assert ids[0] == tok.special_ids["<bos>"] and ids[-1] == tok.special_ids["<eos>"]
    assert tok.decode(ids, skip_specials=True) == "hi"
