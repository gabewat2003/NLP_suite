# Every target runs from a clean clone on fixed seeds (config.SEED / config.SEEDS).
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

.PHONY: data tokenizer baselines embeddings seq2seq eval test smoke all

data:
	$(PY) -m corpus.download
	$(PY) -m corpus.split
	$(PY) -m corpus.general
	$(PY) -m corpus.sources

tokenizer:
	$(PY) -m bpe.scripts.train_bpe --corpus cfpb --vocab-size 16000 --out bpe/artifacts/cfpb_16k
	$(PY) -m bpe.scripts.train_bpe --corpus general --vocab-size 16000 --out bpe/artifacts/general_16k
	$(PY) -m bpe.scripts.compare_general
	$(PY) -m bpe.scripts.eval_tokenizer --tokenizer bpe/artifacts/cfpb_8k --split test
	$(PY) -m bpe.scripts.bench

embeddings:
	$(PY) -m word2vec.scripts.ablate
	$(PY) -m word2vec.scripts.train_w2v
	$(PY) -m word2vec.scripts.crosscheck_gensim
	$(PY) -m word2vec.scripts.neighbors
	$(PY) -m word2vec.scripts.eval_classify

seq2seq:
	$(PY) -m seq2seq.scripts.run_toy
	$(PY) -m seq2seq.scripts.run_taskA
	$(PY) -m seq2seq.scripts.taskA_failures
	$(PY) -m seq2seq.scripts.run_taskB --ablate
	$(PY) -m seq2seq.scripts.run_taskB
	$(PY) -m seq2seq.scripts.plot_attention

baselines:
	$(PY) -m eval.tfidf_baseline

eval:
	$(PY) -m eval.report

test:
	$(PY) -m pytest

smoke:
	$(PY) -m pytest tests/test_end_to_end_smoke.py

all: data tokenizer baselines embeddings seq2seq eval test
