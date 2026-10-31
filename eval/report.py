"""Collect every reports/metrics/*.json into reports/summary.md.

README result tables are copied from this file, never retyped.

    python -m eval.report
"""
import config
from eval.metrics_io import read_metrics


def ms(x, p=3):
    if isinstance(x, dict) and "mean" in x:
        return f"{x['mean']:.{p}f} ± {x['sd']:.{p}f}" if x.get("n", 1) > 1 else f"{x['mean']:.{p}f}"
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def _numeric(cell):
    c = str(cell).replace(",", "").replace("±", " ").replace("x", "").replace("(projected)", "").split()
    try:
        [float(x) for x in c]
        return bool(c)
    except ValueError:
        return False


def table(header, rows):
    right = [all(_numeric(r[i]) for r in rows) if rows else False for i in range(len(header))]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if r else "---" for r in right) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def m(name):
    r = read_metrics(name)
    return r["metrics"] if r else None


def headline():
    rows = []
    cmp_, ev = m("bpe_compare"), m("bpe_eval")
    if cmp_ and ev:
        s8 = {r["trained_on"]: r for r in cmp_["sweep"] if r["vocab_size"] == 8000}
        rows.append(["BPE (CFPB-trained, V=8k)", "Tokens per word on CFPB test", "fertility (lower is better)",
                     ms(s8["cfpb"]["fertility_cfpb"]), f"{ms(s8['general']['fertility_cfpb'])} (WikiText-trained, V=8k); "
                     f"{ms(cmp_['gpt2']['fertility_cfpb'])} (GPT-2, V=50k)"])
        rows.append(["BPE (CFPB-trained, V=8k)", "Tokens per finance term", "term fertility (50 terms)",
                     ms(s8["cfpb"]["term_fertility"], 2), f"{ms(s8['general']['term_fertility'], 2)} (WikiText-trained); "
                     f"{ms(cmp_['gpt2']['term_fertility'], 2)} (GPT-2)"])
        rows.append(["BPE", "Compression on CFPB test", "bytes per token (higher is better)",
                     ms(s8["cfpb"]["bytes_per_token_cfpb"], 2), f"{ms(s8['general']['bytes_per_token_cfpb'], 2)} (WikiText-trained)"])
    wc = m("w2v_classify")
    if wc:
        r = wc["rows"]
        base = f"{ms(r['tfidf']['macro_f1'])} (TF-IDF + LR); {ms(r['random']['macro_f1'])} (random vectors)" if "tfidf" in r else ms(r["random"]["macro_f1"])
        rows.append(["word2vec (SGNS)", "Do mean-pooled vectors classify Product?", "macro-F1, test",
                     ms(r["mine"]["macro_f1"]), base])
    nb = m("w2v_neighbors")
    if nb and "precision_at_5" in nb:
        rows.append(["word2vec", "Do neighbors make domain sense?", "precision@5, 20 probe words (blind-scored)",
                     ms(nb["precision_at_5"]["mine"], 2), f"{ms(nb['precision_at_5']['gensim'], 2)} (gensim)"])
    ta = m("s2s_taskA")
    if ta:
        a = ta["attention"]
        rows.append(["seq2seq (GRU + attention)", "Amount normalization", "exact match: 1-4 digits / 5-7 digits",
                     f"{ms(a['test_in_em'])} / {ms(a['test_out_em'])}",
                     f"{ms(ta['no_attention']['test_in_em'])} / {ms(ta['no_attention']['test_out_em'])} (no attention)"])
    tb = m("s2s_taskB")
    if tb:
        b = tb["beam_trie"]
        base = f"{ms(tb['tfidf']['accuracy'])} / {ms(tb['tfidf']['macro_f1'])} (TF-IDF + LR)" if "tfidf" in tb else "n/a"
        rows.append(["seq2seq", "Narrative -> Product \\| Issue", "accuracy / macro-F1, test (beam 5 + trie)",
                     f"{ms(b['accuracy'])} / {ms(b['macro_f1'])}", base])
    tc = m("s2s_taskB_cls")
    if tc:
        t = tc["test"]
        base = f"{ms(tc['tfidf']['accuracy'])} / {ms(tc['tfidf']['macro_f1'])} (TF-IDF + LR)" if "tfidf" in tc else "n/a"
        rows.append(["seq2seq encoder + classification head", "Narrative -> Product \\| Issue",
                     "accuracy / macro-F1, test", f"{ms(t['accuracy'])} / {ms(t['macro_f1'])}", base])
    return table(["Component", "Headline question", "Metric", "Mine", "Baseline"], rows)


def bpe_section():
    out = []
    cmp_ = m("bpe_compare")
    if cmp_:
        out += ["### BPE: controlled comparison (V = 8k)", ""]
        s8 = {r["trained_on"]: r for r in cmp_["sweep"] if r["vocab_size"] == 8000}
        out += table(["Trained on", "Fertility, CFPB test", "Fertility, WikiText test", "Bytes/token, CFPB test",
                      "Bytes/token, WikiText test", "Finance-term fertility"],
                     [[("CFPB train" if k == "cfpb" else "WikiText sample (size-matched)"),
                       ms(r["fertility_cfpb"]), ms(r["fertility_general"]), ms(r["bytes_per_token_cfpb"], 2),
                       ms(r["bytes_per_token_general"], 2), ms(r["term_fertility"], 2)] for k, r in s8.items()]
                     + [["GPT-2 (V=50,257; not controlled)", ms(cmp_["gpt2"]["fertility_cfpb"]),
                         ms(cmp_["gpt2"]["fertility_general"]), ms(cmp_["gpt2"]["bytes_per_token_cfpb"], 2),
                         ms(cmp_["gpt2"]["bytes_per_token_general"], 2), ms(cmp_["gpt2"]["term_fertility"], 2)]])
        out += ["", "### BPE: vocabulary sweep (fertility on CFPB test / WikiText test)", ""]
        sizes = sorted({r["vocab_size"] for r in cmp_["sweep"]})
        rows = []
        for k in ("cfpb", "general"):
            rr = {r["vocab_size"]: r for r in cmp_["sweep"] if r["trained_on"] == k}
            rows.append([("CFPB-trained" if k == "cfpb" else "WikiText-trained")] +
                        [f"{rr[v]['fertility_cfpb']:.3f} / {rr[v]['fertility_general']:.3f}" for v in sizes])
        out += table(["Trained on"] + [f"V={v // 1000}k" for v in sizes], rows)
        out += ["", "Figure: `reports/figures/bpe_fertility_vs_vocab.png`", "", "### BPE: finance terms at V = 8k", ""]
        out += table(["Term", "CFPB-trained", "WikiText-trained", "GPT-2"],
                     [[f"`{t['term']}`", " ".join(t["cfpb"]), " ".join(t["general"]), " ".join(t["gpt2"])]
                      for t in cmp_["terms_8k"]])
    ev = m("bpe_eval")
    if ev:
        out += ["", "### BPE: CFPB-trained V=8k on CFPB test", ""]
        out += table(["Metric", "Value"], [[k, ms(v, 4)] for k, v in ev.items()])
    be = m("bpe_bench")
    if be:
        out += ["", f"### BPE: trainer speed to V=8000 ({be['hardware']})", ""]
        out += table(["Complaints", "Unique pre-tokens", "Naive (s)", "Incremental (s)", "Speedup"],
                     [[f"{r['docs']:,}", f"{r['unique_pretokens']:,}",
                       f"{r['naive_s']:,.0f}" + (" (projected)" if r["naive_projected"] else ""),
                       f"{r['incremental_s']:.2f}", f"{r['speedup']:.0f}x"] for r in be["rows"]])
        out += ["", f"Encoding: {be['encode_complaints_per_s']:,.0f} complaints/s (pure Python, cold cache). "
                f"Peak RSS {be['peak_rss_mb']:,} MB."]
    return out


def w2v_section():
    out = []
    wc = m("w2v_classify")
    if wc:
        out += ["### word2vec: Product classification on test (mean-pooled vectors + LR)", ""]
        names = {"majority": "Majority class", "random": "Random vectors (same vocab, dim, pooling)",
                 "mine": "My word2vec", "gensim": "gensim word2vec (reference)", "tfidf": "TF-IDF (1-2 grams) + LR"}
        out += table(["Features", "Macro-F1", "Accuracy"],
                     [[names[k], ms(v["macro_f1"]), ms(v["accuracy"])] for k, v in wc["rows"].items()])
        out += ["", "Most frequent confusions (my word2vec, seed 0):", ""]
        out += [f"- {c['true']} -> {c['pred']}: {c['count']}" for c in wc["mine_top_confusions_seed0"]]
    cc = m("w2v_crosscheck")
    if cc:
        out += ["", "### word2vec: cross-check against gensim (mean top-10 neighbor overlap)", ""]
        out += table(["Pair", "Overlap"], [["mine vs gensim (same seed)", ms(cc["mine_vs_gensim"])],
                                           ["gensim vs gensim (seed noise floor)", ms(cc["gensim_vs_gensim"])],
                                           ["mine vs mine (seed noise floor)", ms(cc["mine_vs_mine"])]])
    ab = m("w2v_ablations")
    if ab:
        out += ["", "### word2vec: ablations (val macro-F1, 50k-complaint subset, 3 epochs, 3 seeds)", ""]
        out += table(["Knob", "Value", "Val macro-F1"],
                     [[r["knob"], r["value"], ms(r)] for r in sorted(ab["table"], key=lambda r: -r["mean"])])
        out += ["", f"Adopted for the final model: `{ab['chosen'] or 'base config'}`"]
    nb = m("w2v_neighbors")
    if nb:
        out += ["", "### word2vec: nearest neighbors (my model, seed 0)", ""]
        out += table(["Probe", "Top-5 neighbors"],
                     [[f"`{p}`", ", ".join(w for w, _ in ns)] for p, ns in nb["neighbors_mine"].items()])
        out += ["", "### word2vec: polysemous words (top-10 neighbors)", ""]
        out += table(["Word", "Neighbors"], [[f"`{w}`", ", ".join(x for x, _ in ns)] for w, ns in nb["polysemy"].items()])
        if nb.get("subword"):
            out += ["", f"### word2vec: misspellings (BPE-unit vectors), hit@5 = {ms(nb['subword_hit_at_5'], 2)}", ""]
            out += table(["Misspelled", "Word-level vector?", "BPE-unit top-5", "Correct in top-5"],
                         [[f"`{r['misspelled']}`", "no", ", ".join(r["bpe_top5"]), "yes" if r["hit"] else "no"]
                          for r in nb["subword"]])
    return out


def taskB_figure(tb, tc):
    """Per-seed test accuracy for the two neural models against the TF-IDF and majority lines."""
    from eval import plotstyle
    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    models = [("seq2seq\n(beam 5 + trie)", tb["beam_trie"]["accuracy"], plotstyle.SERIES[0])]
    if tc:
        models.append(("encoder +\nclassification head", tc["test"]["accuracy"], plotstyle.SERIES[2]))
    for y, (name, acc, color) in enumerate(models):
        ax.scatter(acc["values"], [y] * len(acc["values"]), color=color, s=40, alpha=0.8, zorder=3)
        ax.plot([acc["mean"]] * 2, [y - 0.25, y + 0.25], color=color, linewidth=2.5, zorder=3)
    refs = [("TF-IDF + LR", tb["tfidf"]["accuracy"], plotstyle.SERIES[1])] if "tfidf" in tb else []
    for name, x, color in refs:
        ax.axvline(x, color=color, linestyle="--", linewidth=1.5)
        ax.annotate(f"{name} {x:.3f}", (x, len(models) - 0.5), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")
    ax.set_yticks(range(len(models)), [n for n, _, _ in models])
    ax.set_ylim(-0.6, len(models) - 0.3)
    ax.set_xlabel("test exact-match accuracy (dots: seeds; bar: mean)")
    ax.set_title(f"Task B: narrative -> Product | Issue ({tb['n_labels']} labels)")
    return plotstyle.save(fig, "s2s_taskB_models.png")


def s2s_section():
    out = []
    toy = m("s2s_toy")
    if toy:
        out += ["### seq2seq: correctness gates", ""]
        out += table(["Gate", "Result"], [["Copy digits, test exact match", ms(toy["copy_test_em"])],
                                          ["Reverse digits, test exact match", ms(toy["reverse_test_em"])],
                                          ["Reverse: mean attention on the anti-diagonal", ms(toy["reverse_antidiagonal_mean_weight"])],
                                          ["Overfit 32 Task A examples, exact match", ms(toy["overfit32_em"])]])
    ta = m("s2s_taskA")
    if ta:
        out += ["", "### seq2seq Task A: exact match by digit count (train covers 1-4 digits)", ""]
        digits = list(ta["attention"]["em_by_digits"])
        out += table(["Model"] + digits, [[k.replace("_", " ")] + [ms(ta[k]["em_by_digits"][d], 2) for d in digits]
                                          for k in ("attention", "no_attention")])
        out += ["", "Figure: `reports/figures/s2s_taskA_extrapolation.png`; failures: `reports/taskA_failures.md`"]
    tb = m("s2s_taskB")
    if tb:
        c = read_metrics("s2s_taskB")["config"]
        ep = f"{c['epochs']} epochs" if "epochs" in c else f"up to {c['max_epochs']} epochs, early stopping on val"
        out += ["", f"### seq2seq Task B: narrative -> Product | Issue ({tb['n_labels']} labels, test split, "
                f"{ep}, {len(c['seeds'])} seeds)", ""]
        rows = [["Majority label", ms(tb["majority"]["accuracy"]), ms(tb["majority"]["macro_f1"]), "-"]]
        if "tfidf" in tb:
            rows.append(["TF-IDF + LR (predicts label id)", ms(tb["tfidf"]["accuracy"]), ms(tb["tfidf"]["macro_f1"]), "-"])
        for k, name in (("greedy", "Seq2seq, greedy"), ("beam", "Seq2seq, beam 5"), ("beam_trie", "Seq2seq, beam 5 + trie")):
            rows.append([name, ms(tb[k]["accuracy"]), ms(tb[k]["macro_f1"]), ms(tb[k]["invalid_rate"])])
        tc = m("s2s_taskB_cls")
        if tc:
            rows.append(["Seq2seq encoder + classification head", ms(tc["test"]["accuracy"]),
                         ms(tc["test"]["macro_f1"]), "0 (by construction)"])
        out += table(["System", "Exact-match accuracy", "Macro-F1", "Invalid-label rate"], rows)
        h = tb["head_vs_tail_seed0"]
        out += ["", f"Head (10 most frequent labels) accuracy {h['head_acc']:.3f}; tail accuracy {h['tail_acc']:.3f} (seed 0, beam 5 + trie)."]
        taskB_figure(tb, tc)
        out += ["", "Figure: `reports/figures/s2s_taskB_models.png`"]
    ab = m("s2s_taskB_ablations")
    if ab:
        c = read_metrics("s2s_taskB_ablations")["config"]
        out += ["", f"### seq2seq Task B: ablations (val exact match, {c['train_subset'] // 1000}k-complaint subset, "
                f"{c['epochs']} epochs)", ""]
        out += table(["Knob", "Value", "Val exact match", "Seeds"],
                     [[r["knob"], r["value"], ms(r), r["n"]] for r in sorted(ab["table"], key=lambda r: -r["mean"])])
        out += ["", f"Adopted for the final model: `{ab['chosen'] or 'base config'}`"]
    return out


def main():
    lines = ["# Results summary", "", "Generated by `python -m eval.report` from `reports/metrics/*.json`. "
             "Numbers with ± are mean ± sd over seeds (3, or 5 for Task B where stated).", "", "## Headline", ""]
    lines += headline()
    for title, sec in (("BPE tokenizer", bpe_section), ("word2vec", w2v_section), ("seq2seq", s2s_section)):
        body = sec()
        if body:
            lines += ["", f"## {title}", ""] + body
    (config.REPORTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
