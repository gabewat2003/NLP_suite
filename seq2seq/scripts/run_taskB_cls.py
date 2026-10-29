"""Task B with a classification head on the seq2seq encoder (no decoder).

    python -m seq2seq.scripts.run_taskB_cls     # chosen Task B config, full train, TASKB_SEEDS, TEST once

Uses the configuration chosen by the seq2seq ablations (seq2seq/runs/taskB_chosen.json):
same tokenizer, L, word2vec init, hidden and embedding size. Only the output side differs,
so the comparison with the seq2seq decoder isolates "generate the label" vs "score every label".
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import config
from eval.classify import majority, scores, top_confusions
from eval.metrics_io import read_metrics, summarize, write_metrics
from seq2seq.classifier import EncoderClassifier, predict, train_classifier
from seq2seq.common import save_run
from seq2seq.scripts.run_taskB import BASE, FINAL_EPOCHS, PATIENCE, VAL_N, chosen, w2v_init


def build(cfg, seed, limit_train=None):
    from seq2seq.data_taskB import TaskB
    task = TaskB(cfg["tokenizer"], cfg["L"], limit_train)
    label_list = sorted(task.label_ids)
    torch.manual_seed(seed)
    model = EncoderClassifier(task.src_vocab_size, len(label_list), emb=cfg["emb"], hid=cfg["hid"], att=cfg["hid"])
    cover = w2v_init(model, task.tok, cfg["emb"]) if cfg["init"] == "word2vec" else None
    return task, label_list, model, cover


def final_one(args):
    cfg, seed, epochs, patience = args
    torch.set_num_threads(2)
    tag = f"taskB_cls_seed{seed}"
    task, label_list, model, cover = build(cfg, seed)
    idx = {l: i for i, l in enumerate(label_list)}
    y = [idx[l] for l in task.labels["train"]]
    val_src = task.data["val"][:VAL_N]
    val_y = [idx.get(l, -1) for l in task.labels["val"][:VAL_N]]
    model, hist = train_classifier(model, task.data["train"], y, val_src, val_y, epochs=epochs, seed=seed,
                                   patience=patience, log=lambda m: print(f"[{tag}] {m}", flush=True))
    pred = [label_list[i] for i in predict(model, task.data["test"])]
    model.cpu()
    save_run(tag, model, {"cfg": cfg, "seed": seed, "history": hist, "w2v_coverage": cover})
    s = scores(task.labels["test"], pred)
    s.pop("per_class_f1")
    print(f"[{tag}] test acc {s['accuracy']:.4f} macro-F1 {s['macro_f1']:.4f}", flush=True)
    return seed, {"test": s, "history": hist, "preds": pred}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS, help="upper bound; early stop on val")
    ap.add_argument("--patience", type=int, default=PATIENCE, help="0 disables early stopping")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.TASKB_SEEDS))
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    cfg = dict(BASE, **chosen())
    from seq2seq.data_taskB import TaskB, encoded_split
    for sp in ("train", "val", "test"):  # build the cache once, before workers start
        encoded_split(cfg["tokenizer"], sp)
    with ProcessPoolExecutor(min(args.workers, len(seeds))) as ex:
        results = dict(ex.map(final_one, [(cfg, s, args.final_epochs, args.patience or None) for s in seeds]))
    task = TaskB(cfg["tokenizer"], cfg["L"])
    gold, train_lab = task.labels["test"], task.labels["train"]
    res = {"test": {m: summarize([results[s]["test"][m] for s in seeds]) for m in ("accuracy", "macro_f1")},
           "per_seed": {s: results[s]["test"] for s in seeds}}
    pred0 = results[seeds[0]]["preds"]
    freq = {l: c for l, c in zip(*np.unique(train_lab, return_counts=True))}
    head = set(sorted(freq, key=lambda l: -freq[l])[:10])
    res["head_vs_tail_seed0"] = {
        "head_acc": float(np.mean([p == g for p, g in zip(pred0, gold) if g in head])),
        "tail_acc": float(np.mean([p == g for p, g in zip(pred0, gold) if g not in head]))}
    res["top_confusions_seed0"] = top_confusions(gold, pred0, 8)
    res["majority"] = scores(gold, majority(train_lab, len(gold)))
    res["majority"].pop("per_class_f1")
    tf = read_metrics("tfidf_product_issue")
    if tf:
        res["tfidf"] = {k: tf["metrics"]["test"][k] for k in ("accuracy", "macro_f1")}
    res["n_labels"] = len(task.label_ids)
    write_metrics("s2s_taskB_cls", res, {
        "cfg": cfg, "max_epochs": args.final_epochs, "patience": args.patience, "seeds": list(seeds),
        "epochs_run": {s: len(results[s]["history"]) for s in seeds},
        "fit_ids_hash": json.loads((config.PROCESSED / "split_meta.json").read_text())["splits"]["train"]["ids_hash"]})
    print(json.dumps(res["test"], indent=1))


if __name__ == "__main__":
    main()
