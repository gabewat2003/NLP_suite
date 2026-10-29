import torch

from seq2seq.classifier import EncoderClassifier, predict, train_classifier
from seq2seq.train import pad_batch, set_seed
from seq2seq.vocab import PAD


def small_model():
    set_seed(0)
    m = EncoderClassifier(30, 5, emb=8, hid=16, att=16, dropout=0.0)
    m.eval()
    return m


def test_output_shape():
    m = small_model()
    src, src_len = pad_batch([[4, 5, 6, 7], [8, 9]])
    assert m(src, src_len).shape == (2, 5)


def test_padding_invariance():
    m = small_model()
    alone = pad_batch([[9, 10, 11]])
    padded = pad_batch([[9, 10, 11], [4, 5, 6, 7, 8, 9, 10]])
    assert torch.allclose(m(*alone)[0], m(*padded)[0], atol=1e-5)


def test_attention_ignores_padding():
    m = small_model()
    src, src_len = pad_batch([[4, 5, 6, 7, 8], [9, 10]])
    _, alpha = m(src, src_len, return_attn=True)
    assert (alpha[src == PAD] == 0).all()
    assert torch.allclose(alpha.sum(-1), torch.ones(2), atol=1e-6)


def test_overfits_small_set():
    set_seed(0)
    g = torch.Generator().manual_seed(0)
    src = [torch.randint(4, 30, (int(torch.randint(3, 12, (1,), generator=g)),), generator=g).tolist()
           for _ in range(32)]
    y = [i % 5 for i in range(32)]
    m = EncoderClassifier(30, 5, emb=16, hid=32, att=32, dropout=0.0)
    m, hist = train_classifier(m, src, y, src, y, epochs=60, batch_size=32, lr=1e-2, log=lambda *_: None)
    assert hist[-1]["val"] == 1.0 or max(h["val"] for h in hist) == 1.0
    assert predict(m, src) == y
