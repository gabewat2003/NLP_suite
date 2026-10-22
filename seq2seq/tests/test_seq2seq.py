import torch

from seq2seq.data_taskA import make_splits, parse_source, parse_target, source, target
from seq2seq.decode import Trie, beam_search
from seq2seq.model import Seq2Seq
from seq2seq.train import loss_fn, pad_batch, set_seed
from seq2seq.vocab import EOS, PAD


def small_model(attention=True):
    set_seed(0)
    m = Seq2Seq(20, 15, emb=8, hid=16, att=16, dropout=0.0, attention=attention)
    m.eval()
    return m


def batch():
    src = [[4, 5, 6, 7, 8], [9, 10, 11]]
    tgt = [[4, 5, 6], [7, 8]]
    return pad_batch(src, tgt)


def test_no_peeking():
    m = small_model()
    src, src_len, tgt_in, _ = batch()
    base = m(src, src_len, tgt_in)
    t = 1
    changed = tgt_in.clone()
    changed[:, t + 1:] = 13
    out = m(src, src_len, changed)
    assert torch.allclose(base[:, :t + 1], out[:, :t + 1], atol=1e-6)


def test_padding_invariance():
    for att in (True, False):
        m = small_model(att)
        alone = pad_batch([[9, 10, 11]], [[7, 8]])
        padded = pad_batch([[9, 10, 11], [4, 5, 6, 7, 8, 9, 10]], [[7, 8], [4, 5, 6, 7]])
        a = m(alone[0], alone[1], alone[2])
        b = m(padded[0], padded[1], padded[2])
        assert torch.allclose(a[0], b[0, :a.size(1)], atol=1e-5)


def test_attention_mask():
    m = small_model()
    src, src_len, tgt_in, _ = batch()
    _, attn = m(src, src_len, tgt_in, return_attn=True)
    pad = (src == PAD).unsqueeze(1).expand_as(attn)
    assert (attn[pad] == 0).all()
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-6)


def test_loss_ignores_pad():
    m = small_model()
    src, src_len, tgt_in, tgt_out = batch()
    logits = m(src, src_len, tgt_in)
    l1 = loss_fn(logits, tgt_out)
    logits2 = logits.clone()
    logits2[1, 3:] = torch.randn_like(logits2[1, 3:])  # positions past row 1's <eos> are PAD targets
    assert tgt_out[1, 3:].eq(PAD).all()
    assert torch.allclose(l1, loss_fn(logits2, tgt_out))


def test_beam1_equals_greedy():
    m = small_model()
    src, src_len = pad_batch([[4, 5, 6, 7, 8]])
    g = m.greedy(src, src_len, 10)[0].tolist()
    g = g[:g.index(EOS)] if EOS in g else g
    assert beam_search(m, src, src_len, width=1, max_len=10) == g


def test_trie_constrains_output():
    m = small_model()
    src, src_len = pad_batch([[4, 5, 6]])
    valid = [[5, 6, 7], [5, 8], [9]]
    out = beam_search(m, src, src_len, width=3, max_len=10, trie=Trie(valid))
    assert out in valid


def test_taskA_generator_roundtrip():
    for d, c in [(0, 5), (1, 0), (105, 1), (1234, 99), (7_654_321, 10), (1000, 0)]:
        assert parse_source(source(d, c).split()) == (d, c)
        assert parse_target(list(target(d, c))) == (d, c)
    assert source(105, 30) == "one hundred five dollars and thirty cents"
    assert target(250, 30) == "$250.30"


def test_taskA_no_split_overlap():
    sp = make_splits(seed=0, per_digit_train=300, per_digit_eval=50, per_digit_extrap=20)
    keys = {k: {" ".join(s) for s, _, _ in v} for k, v in sp.items()}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not keys[a] & keys[b], (a, b)
    assert {d for _, _, d in sp["test_out"]} == {5, 6, 7}


def test_deterministic_forward():
    a = small_model()
    b = small_model()
    src, src_len, tgt_in, _ = batch()
    assert torch.equal(a(src, src_len, tgt_in), b(src, src_len, tgt_in))


class _StubDecoder:
    """Next-token distribution set by hand: from token t (or <bos>) the next label
    token is t+1; the <eos> logit grows with depth, so the most probable label's
    length is controlled by `slope` and `offset`."""
    V = 16

    def __init__(self, slope, offset):
        self.slope, self.offset = slope, offset
        self.att = None

    def step(self, prev, states, enc_out, mask, final, proj=None):
        depth = states[:, 0]
        logits = torch.zeros(len(prev), self.V)
        nxt = torch.where(prev == 1, torch.full_like(prev, 4), prev + 1).clamp(max=self.V - 1)
        logits[torch.arange(len(prev)), nxt] = 3.0
        logits[:, EOS] = self.slope * depth + self.offset
        return logits, states + 1, None


class _StubModel:
    def __init__(self, slope, offset):
        self.decoder = _StubDecoder(slope, offset)

    def encode(self, src, src_len):
        return torch.zeros(1, 1, 2), torch.zeros(1, 1), torch.zeros(1, 2), torch.ones(1, 1, dtype=torch.bool), None


def _stub_logprob(model, seq):
    states, prev, total = torch.zeros(1, 1), torch.tensor([1]), 0.0
    for t in list(seq) + [EOS]:
        lg, states, _ = model.decoder.step(prev, states, None, None, None)
        total += float(torch.log_softmax(lg, -1)[0, t])
        prev = torch.tensor([t])
    return total


def test_beam_trie_finds_most_probable_label():
    """Nested valid labels ([4], [4,5], ... up to 8 tokens): only one prefix is
    ever live, so a width-2 beam is exact and must return the label with the
    highest total log-prob. A beam that stops once `width` labels have finished
    returns one of the two shortest instead."""
    valid = [list(range(4, 4 + n)) for n in range(1, 9)]
    trie = Trie(valid)
    lengths = set()
    for slope, offset in [(0.5, -2.0), (1.0, -4.0), (1.5, -7.0), (2.0, -12.0), (0.3, 0.0), (3.0, -3.0)]:
        m = _StubModel(slope, offset)
        best = max(valid, key=lambda seq: _stub_logprob(m, seq))
        lengths.add(len(best))
        x, xl = pad_batch([[4, 5, 6]])
        assert beam_search(m, x, xl, width=2, max_len=12, alpha=0.0, trie=trie) == best
    assert max(lengths) > 2  # some optimum is longer than the two shortest labels
