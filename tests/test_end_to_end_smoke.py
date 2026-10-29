"""Whole pipeline on the synthetic fixture: split -> BPE -> word2vec -> seq2seq Task B.
Tiny sizes; checks that the pieces connect, not that they are good."""
import numpy as np

import config


def test_pipeline(processed_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED", processed_dir)
    monkeypatch.setattr(config, "MIN_LABEL_COUNT", 5)

    # BPE on train, evaluated on test
    from bpe.evaluate import all_metrics, load_terms
    from bpe.scripts.train_bpe import load_corpus
    from bpe.tokenizer import Tokenizer
    texts, meta = load_corpus("cfpb")
    tok = Tokenizer.train(texts, 500, meta=meta)
    tok_dir = tok.save(tmp_path / "bpe")
    from corpus.data import load_split
    m = all_metrics(tok, list(load_split("test")["narrative"]), load_terms())
    assert m["unk_rate"] == 0 and m["redaction_integrity"] == 1.0 and m["fertility"] > 0

    # word2vec on train, Product probe on test
    from word2vec.experiment import classify, fit, load_tokens
    tr_df, tr = load_tokens("train")
    te_df, te = load_tokens("test")
    emb, hist = fit({"epochs": 1, "dim": 16, "min_count": 1}, 0, tr, tr_df["complaint_id"])
    s, _ = classify(emb, tr, tr_df["product"], te, te_df["product"])
    assert 0 <= s["macro_f1"] <= 1 and np.isfinite(hist[-1][1])

    # seq2seq Task B with the fixture tokenizer
    import seq2seq.data_taskB as tb
    monkeypatch.setitem(tb.TOKENIZERS, "cfpb", tok_dir)
    from seq2seq.decode import Trie
    from seq2seq.model import Seq2Seq
    from seq2seq.scripts.run_taskB import predict_labels
    from seq2seq.train import train
    task = tb.TaskB("cfpb", L=32)
    model = Seq2Seq(task.src_vocab_size, len(task.tgt_vocab), emb=16, hid=32, att=32)
    model, _ = train(model, task.pairs("train"), task.pairs("val")[:20], epochs=1, batch_size=32, max_steps=5,
                     log=lambda *_: None)
    preds = predict_labels(task, model, task.data["test"][:5], "beam_trie", Trie(task.label_ids.values()))
    assert all(p in task.label_ids for p in preds)  # constrained decoding only emits valid labels

    # classification head on the same encoder
    from seq2seq.classifier import EncoderClassifier, predict, train_classifier
    label_list = sorted(task.label_ids)
    idx = {l: i for i, l in enumerate(label_list)}
    clf = EncoderClassifier(task.src_vocab_size, len(label_list), emb=16, hid=32, att=32)
    clf, _ = train_classifier(clf, task.data["train"], [idx[l] for l in task.labels["train"]],
                              task.data["val"][:20], [idx.get(l, -1) for l in task.labels["val"][:20]],
                              epochs=1, batch_size=32, max_steps=5, log=lambda *_: None)
    assert all(0 <= i < len(label_list) for i in predict(clf, task.data["test"][:5]))
