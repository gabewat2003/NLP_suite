"""Training loop shared by all seq2seq tasks: padding, shifted targets,
ignore_index=PAD, gradient clipping, scheduled sampling, best-on-val checkpoint."""
import copy
import random
import time

import numpy as np
import torch
from torch import nn

from seq2seq.model import Seq2Seq
from seq2seq.vocab import BOS, EOS, PAD

LOG_EVERY = 500  # training steps between progress lines


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device():
    """Training device: NLP_SUITE_DEVICE if set, else CUDA, else Apple-silicon GPU (MPS), else CPU.

    Measured for Task B (batch 64, 128 tokens, hidden 256): 1,488 ms/step on CPU (2 threads) vs
    376 ms on MPS; three MPS processes share the GPU at ~550 ms/step each. Tests and decoding
    use CPU. GPU kernels are not bit-deterministic, so GPU runs can differ slightly across reruns.
    """
    import os
    name = os.environ.get("NLP_SUITE_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pad_batch(src_seqs, tgt_seqs=None, dev=None):
    """src_seqs: lists of ids (no specials). tgt_seqs: lists of ids (no specials).
    Returns src (B,S), src_len (B,), tgt_in = [<bos>, y..] and tgt_out = [y.., <eos>]."""
    B = len(src_seqs)
    S = max(1, max(len(s) for s in src_seqs))
    src = torch.full((B, S), PAD, dtype=torch.long)
    for i, s in enumerate(src_seqs):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    src_len = torch.tensor([max(1, len(s)) for s in src_seqs])
    if tgt_seqs is None:
        return src.to(dev), src_len
    T = max(len(t) for t in tgt_seqs) + 1
    tgt_in = torch.full((B, T), PAD, dtype=torch.long)
    tgt_out = torch.full((B, T), PAD, dtype=torch.long)
    for i, t in enumerate(tgt_seqs):
        tgt_in[i, 0] = BOS
        tgt_in[i, 1:len(t) + 1] = torch.tensor(t, dtype=torch.long)
        tgt_out[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        tgt_out[i, len(t)] = EOS
    return src.to(dev), src_len, tgt_in.to(dev), tgt_out.to(dev)


def loss_fn(logits, tgt_out):
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD)


def greedy_decode(model, src_seqs, max_len, batch_size=256, dev=None):
    """Returns list of id lists (cut at the first <eos>). Runs on the model's device."""
    dev = dev or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(src_seqs), batch_size):
        src, src_len = pad_batch(src_seqs[i:i + batch_size], dev=dev)
        toks = model.greedy(src, src_len, max_len).tolist()
        for t in toks:
            out.append(t[:t.index(EOS)] if EOS in t else t)
    return out


def train(model, train_data, val_data, epochs=10, batch_size=64, lr=1e-3, clip=1.0, tf_ratio=1.0,
          tf_final=None, seed=0, max_len=40, patience=None, log=print, val_fn=None, max_steps=None):
    """train_data / val_data: lists of (src_ids, tgt_ids). Keeps the weights with
    the best val exact match. tf_final: if set, teacher forcing decays linearly
    from tf_ratio to tf_final over training (scheduled sampling)."""
    set_seed(seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    best, best_state, history, bad = -1.0, None, [], 0
    steps_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    step = 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            src, src_len, tgt_in, tgt_out = pad_batch([train_data[j][0] for j in idx],
                                                      [train_data[j][1] for j in idx], dev)
            tf = tf_ratio if tf_final is None else tf_ratio + (tf_final - tf_ratio) * step / max(total_steps, 1)
            loss = loss_fn(model(src, src_len, tgt_in, tf_ratio=tf), tgt_out)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # RNNs: clip or risk exploding gradients
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
            step += 1
            if step % LOG_EVERY == 0:  # epochs are long; show progress so a stall is visible
                log(f"    step {step}/{total_steps}  loss {tot / max(n, 1):.4f}  {time.perf_counter() - t0:.0f}s")
            if max_steps and step >= max_steps:
                break
        if val_fn is not None:
            val = val_fn(model)
        else:
            preds = greedy_decode(model, [s for s, _ in val_data], max_len, dev=dev)
            val = float(np.mean([p == t for p, (_, t) in zip(preds, val_data)]))
        history.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "val": val,
                        "seconds": round(time.perf_counter() - t0, 1)})
        log(f"  epoch {ep + 1}/{epochs}  loss {tot / max(n, 1):.4f}  val {val:.4f}  {time.perf_counter() - t0:.0f}s")
        if val > best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if patience and bad >= patience:
                break
        if max_steps and step >= max_steps:
            break
    model.load_state_dict(best_state)
    return model, history


def build(src_vocab, tgt_vocab, **kw):
    return Seq2Seq(src_vocab, tgt_vocab, **kw)
