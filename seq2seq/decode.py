"""Beam search (with length normalization) and trie-constrained decoding.

Greedy decoding lives on Seq2Seq.greedy. Beam width 1 reproduces greedy exactly.
"""
import torch

from seq2seq.vocab import BOS, EOS


class Trie:
    """Valid output sequences (each ends with <eos>); allowed(prefix) gives legal next ids."""

    def __init__(self, sequences):
        self.root = {}
        for seq in sequences:
            node = self.root
            for t in list(seq) + [EOS]:
                node = node.setdefault(t, {})

    def allowed(self, prefix):
        node = self.root
        for t in prefix:
            node = node.get(t)
            if node is None:
                return []
        return list(node)


@torch.no_grad()
def beam_search(model, src, src_len, width=5, max_len=40, alpha=1.0, trie=None):
    """Decode one example (src: (1, S)).

    Hypotheses are scored by summed log-prob; finished ones are ranked by
    score / len**alpha. Without length normalization (alpha=0) beam search
    prefers short outputs, since every extra token adds a negative log-prob.

    Each step keeps the `width` best live hypotheses; candidates ending in <eos>
    move to `finished`. Search continues until no live hypothesis could still
    win: for alpha=0 that is when the best finished score beats the best live
    score (live scores only fall); otherwise when nothing is live or max_len.
    (An earlier version stopped once `width` hypotheses had finished, which let
    early, unlikely completions crowd out the best one still being extended.)
    """
    enc_out, s, final, mask, proj = model.encode(src, src_len)
    live = [(0.0, [], s[0])]
    finished = []
    norm = lambda score, n: score / (n ** alpha) if alpha else score
    for _ in range(max_len):
        n = len(live)
        inp = torch.tensor([h[-1] if h else BOS for _, h, _ in live], device=src.device)
        states = torch.stack([st for _, _, st in live])
        lg, new_s, _ = model.decoder.step(inp, states, enc_out.expand(n, -1, -1), mask.expand(n, -1),
                                          final.expand(n, -1), None if proj is None else proj.expand(n, -1, -1))
        logp = torch.log_softmax(lg, -1)
        if trie is not None:
            m = torch.full_like(logp, float("-inf"))
            for i, (_, h, _) in enumerate(live):
                ok = trie.allowed(h)
                if ok:
                    m[i, ok] = 0.0
            logp = logp + m
        cand = []
        k = min(width, logp.size(1))
        top_v, top_i = logp.topk(k, dim=-1)
        for i, (score, h, _) in enumerate(live):
            for v, t in zip(top_v[i].tolist(), top_i[i].tolist()):
                if v != float("-inf"):
                    cand.append((score + v, h + [t], new_s[i]))
        cand.sort(key=lambda c: -c[0])
        live = []
        for score, h, st in cand[:width]:
            if h[-1] == EOS:
                finished.append((norm(score, len(h)), h[:-1]))
            else:
                live.append((score, h, st))
        finished = sorted(finished, key=lambda f: -f[0])[:width]
        if not live:
            break
        if alpha == 0 and finished and finished[0][0] >= max(sc for sc, _, _ in live):
            break
    if not finished:
        finished = [(norm(sc, len(h)), h) for sc, h, _ in live]
    return max(finished, key=lambda f: f[0])[1]
