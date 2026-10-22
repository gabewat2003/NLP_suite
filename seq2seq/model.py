"""GRU encoder-decoder with additive (Bahdanau) attention, written from nn.GRU /
nn.GRUCell / nn.Linear only (no nn.Transformer, no seq2seq library)."""
import torch
from torch import nn

from seq2seq.vocab import BOS, PAD


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid, pad_id=PAD, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, hid)  # fwd+bwd final state -> decoder init
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))                                            # (B, S, E)
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))  # (B,S,2H)
        final = torch.cat([h[0], h[1]], dim=-1)                                 # (B, 2H)
        return out, torch.tanh(self.bridge(final)), final


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, att_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, att_dim, bias=True)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, s, enc_out, mask, enc_proj=None):
        # s: (B, H)  enc_out: (B, S, 2H)  mask: (B, S) True on real tokens
        proj = self.W_h(enc_out) if enc_proj is None else enc_proj
        scores = self.v(torch.tanh(proj + self.W_s(s).unsqueeze(1))).squeeze(-1)  # (B, S)
        scores = scores.masked_fill(~mask, float("-inf"))     # padding gets exactly zero weight
        alpha = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)  # (B, 2H)
        return ctx, alpha


class Decoder(nn.Module):
    def __init__(self, vocab, emb, hid, enc_dim, att_dim, pad_id=PAD, dropout=0.2, attention=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_id)
        self.att = Attention(enc_dim, hid, att_dim) if attention else None
        self.cell = nn.GRUCell(emb + enc_dim, hid)
        self.out = nn.Linear(hid + enc_dim + emb, vocab)
        self.drop = nn.Dropout(dropout)

    def step(self, prev_tok, s, enc_out, mask, enc_final, enc_proj=None):
        e = self.drop(self.emb(prev_tok))                                       # (B, E)
        if self.att is not None:
            ctx, alpha = self.att(s, enc_out, mask, enc_proj)                   # attend with previous state
        else:  # no attention: the same fixed summary vector at every step
            ctx, alpha = enc_final, None
        s = self.cell(torch.cat([e, ctx], dim=-1), s)                          # (B, H)
        logits = self.out(torch.cat([s, ctx, e], dim=-1))                      # (B, V)
        return logits, s, alpha


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=128, hid=256, att=256, dropout=0.2, attention=True):
        super().__init__()
        self.hparams = dict(src_vocab=src_vocab, tgt_vocab=tgt_vocab, emb=emb, hid=hid, att=att,
                            dropout=dropout, attention=attention)
        self.encoder = Encoder(src_vocab, emb, hid, dropout=dropout)
        self.decoder = Decoder(tgt_vocab, emb, hid, 2 * hid, att, dropout=dropout, attention=attention)

    def encode(self, src, src_len):
        enc_out, s, final = self.encoder(src, src_len)
        mask = src != PAD
        proj = self.decoder.att.W_h(enc_out) if self.decoder.att is not None else None
        return enc_out, s, final, mask, proj

    def forward(self, src, src_len, tgt_in, tf_ratio=1.0, return_attn=False):
        """tgt_in = [<bos>, y1 .. yn] (padded). Returns logits (B, T, V) for targets
        [y1 .. yn, <eos>]. With tf_ratio < 1 the model's own argmax is fed back
        (scheduled sampling)."""
        enc_out, s, final, mask, proj = self.encode(src, src_len)
        T = tgt_in.size(1)
        inp = tgt_in[:, 0]
        logits, attns = [], []
        for t in range(T):
            lg, s, a = self.decoder.step(inp, s, enc_out, mask, final, proj)
            logits.append(lg)
            if return_attn:
                attns.append(a)
            if t + 1 < T:
                if tf_ratio >= 1.0 or torch.rand(()) < tf_ratio:
                    inp = tgt_in[:, t + 1]
                else:
                    inp = lg.argmax(-1)
        out = torch.stack(logits, 1)
        if return_attn:
            return out, (torch.stack(attns, 1) if attns[0] is not None else None)
        return out

    @torch.no_grad()
    def greedy(self, src, src_len, max_len, return_attn=False):
        enc_out, s, final, mask, proj = self.encode(src, src_len)
        B = src.size(0)
        inp = torch.full((B,), BOS, dtype=torch.long, device=src.device)
        toks, attns = [], []
        for _ in range(max_len):
            lg, s, a = self.decoder.step(inp, s, enc_out, mask, final, proj)
            inp = lg.argmax(-1)
            toks.append(inp)
            attns.append(a)
        out = torch.stack(toks, 1)
        if return_attn:
            return out, (torch.stack(attns, 1) if attns[0] is not None else None)
        return out
