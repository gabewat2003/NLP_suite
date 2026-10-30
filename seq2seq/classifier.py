"""Task B without a decoder: the seq2seq encoder plus a classification head.

Same bidirectional GRU encoder as seq2seq/model.py (so word2vec init, tokenizer and
L carry over unchanged). Instead of generating the label token by token, an additive
attention pool summarizes the encoder states, and one linear layer scores all labels
at once. No decoding, no beam search, no invalid outputs.
"""
import copy
import random
import time

import numpy as np
import torch
from torch import nn

from seq2seq.model import Encoder
from seq2seq.train import LOG_EVERY, device, pad_batch, set_seed
from seq2seq.vocab import PAD


class EncoderClassifier(nn.Module):
    def __init__(self, src_vocab, n_labels, emb=128, hid=256, att=256, dropout=0.2):
        super().__init__()
        self.hparams = dict(src_vocab=src_vocab, n_labels=n_labels, emb=emb, hid=hid, att=att, dropout=dropout)
        self.encoder = Encoder(src_vocab, emb, hid, dropout=dropout)
        self.W = nn.Linear(2 * hid, att)
        self.v = nn.Linear(att, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        # input: attention-pooled states (2H) + final fwd/bwd state (2H)
        self.out = nn.Linear(4 * hid, n_labels)

    def forward(self, src, src_len, return_attn=False):
        enc_out, _, final = self.encoder(src, src_len)                    # (B,S,2H), (B,2H)
        mask = src != PAD
        scores = self.v(torch.tanh(self.W(enc_out))).squeeze(-1)          # (B, S)
        alpha = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        ctx = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)           # (B, 2H)
        logits = self.out(self.drop(torch.cat([ctx, final], dim=-1)))     # (B, n_labels)
        return (logits, alpha) if return_attn else logits


@torch.no_grad()
def predict(model, src_seqs, batch_size=256, dev=None):
    """Label index per source sequence (argmax)."""
    dev = dev or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(src_seqs), batch_size):
        src, src_len = pad_batch(src_seqs[i:i + batch_size], dev=dev)
        out.extend(model(src, src_len).argmax(-1).tolist())
    return out


def train_classifier(model, src, y, val_src, val_y, epochs=10, batch_size=64, lr=1e-3, clip=1.0, seed=0,
                     patience=None, log=print, max_steps=None):
    """src: lists of ids; y: label indices. Keeps the weights with the best val accuracy."""
    set_seed(seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    best, best_state, history, bad, step = -1.0, None, [], 0, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = list(range(len(src)))
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            x, xl = pad_batch([src[j] for j in idx], dev=dev)
            t = torch.tensor([y[j] for j in idx], device=dev)
            loss = nn.functional.cross_entropy(model(x, xl), t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
            step += 1
            if step % LOG_EVERY == 0:
                log(f"    step {step}  loss {tot / max(n, 1):.4f}  {time.perf_counter() - t0:.0f}s")
            if max_steps and step >= max_steps:
                break
        val = float(np.mean(np.array(predict(model, val_src, dev=dev)) == np.array(val_y)))
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
