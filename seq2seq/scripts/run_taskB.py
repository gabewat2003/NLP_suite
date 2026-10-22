"""Task B: narrative -> "Product | Issue" label string.

    python -m seq2seq.scripts.run_taskB --ablate   # one-knob ablations on VAL (50k train subset, 3 seeds)
    python -m seq2seq.scripts.run_taskB            # chosen config, full train, 3 seeds, TEST once

Test decoding: greedy, beam 5, beam 5 + trie constraint (only valid labels).
Compared against majority and TF-IDF + LR on the same labels.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import config
from eval.classify import majority, scores, top_confusions
from eval.labels import OTHER
from eval.metrics_io import read_metrics, summarize, write_metrics
from seq2seq.common import RUNS, save_run
from seq2seq.decode import Trie, beam_search
from seq2seq.model import Seq2Seq
from seq2seq.train import greedy_decode, pad_batch, train
from word2vec.experiment import ART as W2V_ART

# emb matches the chosen word2vec dimension so the word2vec-init ablation changes only the init
BASE = dict(tokenizer="cfpb", L=128, attention=True, init="random", tf_final=None, hid=256,
            emb=json.loads((W2V_ART / "chosen_config.json").read_text())["config"].get("dim", 100)
            if (W2V_ART / "chosen_config.json").exists() else 100)
ABLATIONS = [("attention", False), ("L", 64), ("L", 256), ("init", "word2vec"), ("tokenizer", "general"),
             ("tf_final", 0.5)]
# Reduced compute budget (CPU only): hidden-size ablations dropped, 2-epoch ablations, 4-epoch final runs.
ABL_TRAIN, ABL_EPOCHS, FINAL_EPOCHS, VAL_N = 50_000, 2, 4, 4000
# Ablations are for choosing a config only: the base config gets 3 seeds (its sd sets the
# adoption threshold), each knob 1 seed. Final runs use all 3 seeds.
ABL_KNOB_SEEDS = (0,)
INVALID = "<invalid>"
# Beam ranking for Task B: total log-prob (no length normalization). Picking a label means
# wanting argmax P(label); chosen on held-out val (0.453 vs 0.430 with per-token normalization).
BEAM_ALPHA = 0.0


def w2v_init(model, tok, emb_dim):
    from word2vec.model import Embeddings
    e = Embeddings.load(W2V_ART / "bpe_seed0")
    if e.matrix.shape[1] != emb_dim:
        raise ValueError("emb dim must match word2vec dim for init")
    W = model.encoder.emb.weight.data
    mat = e.matrix / e.matrix.std()  # match nn.Embedding's N(0, 1) scale
    hit = 0
    for i in range(len(tok)):
        j = e.index.get(tok.token_text(i))
        if j is not None:
            W[i] = torch.from_numpy(mat[j])
            hit += 1
    return hit / len(tok)


def predict_labels(task, model, src, mode, trie=None):
    if mode == "greedy":
        return [task.to_label(p) or INVALID for p in greedy_decode(model, src, 30)]
    out = []
    model.eval()
    dev = next(model.parameters()).device
    for s in src:
        x, xl = pad_batch([s], dev=dev)
        ids = beam_search(model, x, xl, width=5, max_len=30, alpha=BEAM_ALPHA,
                          trie=trie if mode == "beam_trie" else None)
        out.append(task.to_label(ids) or INVALID)
    return out


def build(cfg, seed, limit_train):
    from seq2seq.data_taskB import TaskB
    task = TaskB(cfg["tokenizer"], cfg["L"], limit_train)
    torch.manual_seed(seed)
    model = Seq2Seq(task.src_vocab_size, len(task.tgt_vocab), emb=cfg["emb"], hid=cfg["hid"], att=cfg["hid"],
                    attention=cfg["attention"])
    cover = w2v_init(model, task.tok, cfg["emb"]) if cfg["init"] == "word2vec" else None
    return task, model, cover


def fit(cfg, seed, limit_train, epochs, tag):
    torch.set_num_threads(2)
    task, model, cover = build(cfg, seed, limit_train)
    val_src, val_lab = task.data["val"][:VAL_N], task.labels["val"][:VAL_N]
    val_fn = lambda m: float(np.mean([p == g for p, g in zip(predict_labels(task, m, val_src, "greedy"), val_lab)]))
    model, hist = train(model, task.pairs("train"), None, epochs=epochs, batch_size=64, seed=seed,
                        tf_final=cfg["tf_final"], val_fn=val_fn, log=lambda m: print(f"[{tag}] {m}", flush=True))
    return task, model, hist, cover


def ablate_one(args):
    torch.set_num_threads(2)
    knob, value, seed = args
    cfg = dict(BASE, **({knob: value} if knob else {}))
    tag = f"abl_{knob or 'base'}_{value}_s{seed}"
    _, _, hist, _ = fit(cfg, seed, ABL_TRAIN, ABL_EPOCHS, tag)
    return knob, value, seed, max(h["val"] for h in hist), hist[-1]["seconds"]


def evaluate_seed(task, model, seed, hist=None):
    """Test-set decoding for one trained model: greedy, beam 5, beam 5 + trie."""
    tag = f"taskB_seed{seed}"
    model.cpu()  # beam search runs one example at a time: cheaper on CPU than many tiny GPU launches
    trie = Trie(task.label_ids.values())
    gold = task.labels["test"]
    out = {"history": hist}
    for mode in ("greedy", "beam", "beam_trie"):
        pred = predict_labels(task, model, task.data["test"], mode, trie)
        out[mode] = {**scores(gold, pred), "invalid_rate": float(np.mean([p == INVALID for p in pred]))}
        out[mode].pop("per_class_f1")
        if mode == "beam_trie":
            out["preds_beam_trie"] = pred
        print(f"[{tag}] {mode}: acc {out[mode]['accuracy']:.4f} macro-F1 {out[mode]['macro_f1']:.4f} "
              f"invalid {out[mode]['invalid_rate']:.4f}", flush=True)
    return out


def final_one(args):
    cfg, seed = args
    tag = f"taskB_seed{seed}"
    task, model, hist, cover = fit(cfg, seed, None, FINAL_EPOCHS, tag)
    model.cpu()
    save_run(tag, model, {"cfg": cfg, "seed": seed, "history": hist, "w2v_coverage": cover})
    return seed, evaluate_seed(task, model, seed, hist)


def decode_one(seed):
    """Re-run test decoding for a saved model (no retraining)."""
    torch.set_num_threads(3)
    from seq2seq.data_taskB import TaskB
    run = json.loads((RUNS / f"taskB_seed{seed}" / "run.json").read_text())
    cfg, hp = run["cfg"], run["hparams"]
    task = TaskB(cfg["tokenizer"], cfg["L"])
    model = Seq2Seq(hp["src_vocab"], hp["tgt_vocab"], emb=hp["emb"], hid=hp["hid"], att=hp["att"],
                    dropout=hp["dropout"], attention=hp["attention"])
    model.load_state_dict(torch.load(RUNS / f"taskB_seed{seed}" / "model.pt"))
    return seed, evaluate_seed(task, model, seed, run["history"])


def parse_log(path):
    """Best val per finished ablation run in an earlier log: {(knob, value, seed): best_val}."""
    import re
    vals, finished = {}, set()
    pat = re.compile(r"\[abl_(\w+?)_(.+)_s(\d+)\]\s+epoch (\d+)/(\d+)\s+loss \S+\s+val (\S+)")
    for line in open(path):
        m = pat.search(line)
        if not m:
            continue
        knob, value, seed, ep, n_ep, val = m.groups()
        knob = None if knob == "base" else knob
        match = [(k, v) for k, v in [(None, None)] + ABLATIONS if k == knob and str(v) == value]
        if not match:
            continue
        key = (match[0][0], match[0][1], int(seed))
        vals[key] = max(vals.get(key, 0.0), float(val))
        if ep == n_ep:
            finished.add(key)
    return {k: v for k, v in vals.items() if k in finished}


def chosen():
    p = RUNS / "taskB_chosen.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume-log", help="reuse ablation runs that finished in an earlier (interrupted) log")
    ap.add_argument("--decode-only", action="store_true", help="re-run test decoding for saved final models")
    args = ap.parse_args()
    from seq2seq.data_taskB import encoded_split
    for tk in ("cfpb", "general"):  # build caches once, before workers start
        for sp in ("train", "val", "test"):
            encoded_split(tk, sp)
    if args.ablate:
        jobs = [(None, None, s) for s in config.SEEDS] + [(k, v, s) for k, v in ABLATIONS for s in ABL_KNOB_SEEDS]
        done = parse_log(args.resume_log) if args.resume_log else {}
        res = [(k, v, s, done[(k, v, s)], None) for k, v, s in jobs if (k, v, s) in done]
        todo = [j for j in jobs if j not in done]
        print(f"reusing {len(res)} finished runs, running {len(todo)}", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            res += list(ex.map(ablate_one, todo))
        rows = {}
        for k, v, s, val, secs in res:
            rows.setdefault((k, v), []).append(val)
        table = [{"knob": k or "base", "value": v, **summarize(x)} for (k, v), x in rows.items()]
        base = next(r for r in table if r["knob"] == "base")
        pick = {}
        for r in table:
            if r["knob"] != "base" and r["mean"] > base["mean"] + max(base["sd"], 1e-9):
                if r["knob"] not in pick or r["mean"] > pick[r["knob"]][1]:
                    pick[r["knob"]] = (r["value"], r["mean"])
        cfg = {k: v for k, (v, _) in pick.items()}
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "taskB_chosen.json").write_text(json.dumps(cfg))
        write_metrics("s2s_taskB_ablations", {"table": table, "chosen": cfg},
                      {"train_subset": ABL_TRAIN, "epochs": ABL_EPOCHS, "val_n": VAL_N, "metric": "val exact match",
                       "base_seeds": len(config.SEEDS), "knob_seeds": len(ABL_KNOB_SEEDS),
                       "base": BASE})
        for r in sorted(table, key=lambda r: -r["mean"]):
            print(f"{r['knob']:>10} {str(r['value']):>8}  {r['mean']:.4f} ± {r['sd']:.4f}")
        print("chosen:", cfg)
        return
    cfg = dict(BASE, **chosen())
    with ProcessPoolExecutor(min(args.workers, len(config.SEEDS))) as ex:
        if args.decode_only:
            results = dict(ex.map(decode_one, config.SEEDS))
        else:
            results = dict(ex.map(final_one, [(cfg, s) for s in config.SEEDS]))
    from seq2seq.data_taskB import TaskB
    task = TaskB(cfg["tokenizer"], cfg["L"])
    gold, train_lab = task.labels["test"], task.labels["train"]
    res = {}
    for mode in ("greedy", "beam", "beam_trie"):
        res[mode] = {m: summarize([results[s][mode][m] for s in config.SEEDS])
                     for m in ("accuracy", "macro_f1", "invalid_rate")}
    pred0 = results[config.SEEDS[0]]["preds_beam_trie"]
    freq = {l: c for l, c in zip(*np.unique(train_lab, return_counts=True))}
    head = set(sorted(freq, key=lambda l: -freq[l])[:10])
    res["head_vs_tail_seed0"] = {
        "head_acc": float(np.mean([p == g for p, g in zip(pred0, gold) if g in head])),
        "tail_acc": float(np.mean([p == g for p, g in zip(pred0, gold) if g not in head])),
        "head_labels": sorted(head)}
    res["top_confusions_seed0"] = top_confusions(gold, pred0, 8)
    res["majority"] = scores(gold, majority(train_lab, len(gold)))
    res["majority"].pop("per_class_f1")
    tf = read_metrics("tfidf_product_issue")
    if tf:
        res["tfidf"] = {k: tf["metrics"]["test"][k] for k in ("accuracy", "macro_f1")}
    res["n_labels"] = len(task.label_ids)
    res["other_share_test"] = float(np.mean([g == OTHER for g in gold]))
    (RUNS / "taskB_test_preds_seed0.json").write_text(json.dumps(pred0))
    write_metrics("s2s_taskB", res, {"cfg": cfg, "epochs": FINAL_EPOCHS, "seeds": list(config.SEEDS),
                                      "beam_alpha": BEAM_ALPHA,
                                      "fit_ids_hash": json.loads((config.PROCESSED / "split_meta.json").read_text())
                                      ["splits"]["train"]["ids_hash"]})
    print(json.dumps({k: v for k, v in res.items() if k != "head_vs_tail_seed0"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
