# Lab notes

Observations recorded while running the pipeline, in the order they came up. The component READMEs are written from these; every number here is also in `reports/summary.md` or the metrics files.

## Data
- The CFPB removed complaint narratives from its public database in September 2026 (Release 24). The bulk
  export now has no narrative column, and the old search API returns "Not found". We use a CC0 Hugging Face
  mirror captured before the removal, pinned to one revision and checked against the published SHA-256s.
- Product names change in April 2017 and again in 2023 (credit reporting renamed, "Credit card or prepaid
  card" split back into "Credit card" / "Prepaid card", payday loan renamed). 2018-2022 is the only window
  where every Product label is constant.
- 29% of narratives in the window are duplicates: 191,793 exact copies and 85,366 near-copies (same first
  200 characters). Most are template dispute letters submitted many times. Without dedup they would sit in
  both train and test.
- Redaction normalization: 776k `<REDACTED>`, 275k `<DATE>`, 147k `<AMT>` in train. A first pass missed
  amounts that were themselves masked (`{$XXXX}`, 72 cases) and space-damaged ones (`{$13 0.00}`); the amount
  pattern now covers those. Remaining `XX` hits are Roman numerals (`Title XXIII`).
- Label shift across the time split: credit reporting is 39.3% of train (2018-2021) but 51.5% of test
  (2H 2022); debt collection drops from 20.4% to 12.4%. A random split would hide this.

## BPE
- Training is fast once pre-tokens are counted: 165 MB of complaints to a 16k vocab in 25 s (incremental
  trainer). WikiText (same byte count, 372k distinct pre-tokens vs 104k for complaints) takes 48 s.
- The controlled 2x2 shows the expected diagonal: each tokenizer is best on its own domain at every vocab size.
  At 8k, CFPB-trained fertility on CFPB test is 1.206 vs 1.484 for WikiText-trained (19% fewer tokens), and
  it even beats GPT-2's 50k vocab (1.240) on complaint text. The price: 1.646 vs 1.334 on WikiText.
- Finance terms: CFPB-trained 1.16 tokens/term vs 2.74 WikiText-trained vs 1.58 GPT-2.
  `forbearance` -> WikiText-trained `for b ear ance` (it finds "for" and "bear"); GPT-2 `forb earance`.
  `Equifax` -> `E qu if ax`; `FDCPA` -> `F D C PA`.
- Merges that are artifacts of the data: `▁1681` is a single token, because template dispute letters cite
  "15 U.S.C. 1681" (the FCRA). A 32-underscore token and a 16-asterisk token come from form lines pasted into
  complaints. Company names (`▁Navient`, `▁Ocwen`, `▁Comenity`, `▁Synchrony`) are whole tokens.
- `charge-off` stays 3 pieces in every tokenizer: the GPT-2 pre-tokenizer splits on the hyphen, so no merge
  can cross it.
- Naive vs incremental trainer: identical merge lists. First benchmark (CPU shared with other jobs): 247x / 354x / 222x
  at 1k / 10k / 100k. Rerun with the CPU otherwise idle: 200x / 263x / 208x. The README uses the rerun.
- 100k complaints (rerun): incremental 9.1 s vs naive ~32 min (projected from its first 1,226 merges). The projection
  uses the earliest merges, which are the naive trainer's slowest (words are longest then), so it overstates
  the naive time somewhat. Encoding: ~3.5k complaints/s in pure Python with the pre-token cache.
- Quirk: special tokens are split out before pre-tokenization, so the space in front of a redaction token is
  left as its own `▁` token ("charged <AMT>" -> `▁charged ▁ <AMT>`). Every redaction costs 2 tokens. Folding
  the space into the special (or stripping it) would shave this; left as is and documented.
- Demo sentence (synthetic): 15 tokens with the CFPB tokenizer vs 28 with the WikiText one.

## seq2seq: toy gates and Task A
- Toy gates: copy 100%, reverse 99.9% (mean attention weight on the anti-diagonal 0.80), overfit-32 100%
  (loss 0.0002). The reverse heatmap is a clean anti-diagonal.
- Task A: 1-4 digits ~100% exact match with or without attention; 5, 6 and 7 digits: 0.0% for every seed,
  both variants. Every one of the 900 extrapolation outputs has exactly 4 dollar digits. Cents are right 100%
  of the time and the last three dollar digits 99.2%: the model learned the output template `$X,XXX.XX` and
  fills it from the right, dropping leading digits ("sixty thousand eight hundred eighty nine" -> `$6,889`).
  `million` (unseen) doesn't change the failure mode: 7-digit errors look the same as 5- and 6-digit ones.
- Attention doesn't matter on Task A in range: inputs are ~6-12 words, short enough for the final state.

## word2vec
- Ablations (50k subset, 3 epochs, val): dim 300 (0.701) and window 10 (0.699) beat base 0.686 by >1 sd; adopted.
  BPE units were worst (0.661).
- Final (full train, 5 epochs, 3 seeds): 22,948-word vocab, 28.6M tokens, ~126M pairs/epoch, ~80 min/seed.
- Product macro-F1 on test: mine 0.736 ± 0.004, gensim 0.732, random 0.607, TF-IDF 0.757.
- gensim neighbor overlap 0.69 vs noise floors 0.72 (mine) / 0.75 (gensim): close but below. Likely batching.
- Precision@5 (blind, scored by Claude) mine 0.93, gensim 0.97, after correcting `athens` (a collection agency).
- `balance` neighbors are pasted credit-report fragments (`statusclosed`, `typesecured`, `updatedxxxx`).
- Subword misspelling test: 1/41. Domain BPE makes the correct terms atomic, so misspellings share no pieces.

## Baselines
- TF-IDF Product | Issue: the warm-started C=3 fit stopped after 0 iterations (its val score repeats C=1);
  C=10 was selected and fit normally (326 iterations).

## seq2seq: Task B
- Why Product | Issue accuracy is ~48% (TF-IDF): 56-way exact match; majority 18.8%; Product alone is 84.3%.
  TF-IDF's top four confusions all keep the product right and swap a neighboring credit-reporting issue
  (Improper use / Incorrect information / company's investigation). And drift: "Improper use of your report"
  is 4.1% of train, 15.6% of val, 17.5% of test (surge in 2022); its confusion with "Incorrect information"
  is the single largest error (1,962 test complaints).
- Task B ablations (val exact match, 50k complaints, 2 epochs; base 3 seeds 0.436 ± 0.006, knobs 1 seed):
  word2vec init 0.477 (adopted), L=256 0.448 (adopted), no attention 0.440 (not adopted, within noise),
  scheduled sampling 0.432, L=64 0.417, WikiText tokenizer 0.378. The two cross-component links both
  matter: word2vec init +4 points, domain BPE vs general BPE +6 points.
- Final runs use the configuration the adoption rule picked (L=256 + word2vec init): 4 epochs, 3 seeds sharing
  the Apple GPU, 2.5-3.7 h per seed.
- BUG found and fixed in beam search: it stopped once `width` hypotheses had finished, so early, unlikely
  completions crowded out the best hypothesis still being extended. On seed 1's test set, beam 5 scored 0.281
  vs greedy 0.479. Fixed: keep `width` live hypotheses, keep searching until no live one can win. The old unit
  test (beam width 1 == greedy) could not trigger it; a new test with a hand-built stub model, where the most
  probable label is known exactly, fails on the old decoder and passes on the fixed one.
- Beam ranking for Task B: total log-prob (alpha=0), chosen on held-out val (last 1,000 val complaints):
  0.453 vs 0.430 with per-token normalization. Test decoding re-run from the saved models with the fix.
- Final test result (beam 5, total log-prob): accuracy 0.513 ± 0.039 vs TF-IDF 0.483; macro-F1 0.363 ± 0.014 vs
  0.365. Seed spread (0.483-0.557) is as large as the gap. Attention concentrates on evidence words (`furnish`
  38x its frequency, `erroneous`, `consent`) and gives redaction tokens a fifth of their share.
