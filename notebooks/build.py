"""Build and execute the demo notebooks so their saved outputs match the code.

    python notebooks/build.py            # all
    python notebooks/build.py bpe        # one

Notebooks are exploration only: no logic lives in them, and no complaint text is
printed (only tokens, labels and synthetic sentences).
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SETUP = "import sys, os\nos.chdir(r'{root}')\nsys.path.insert(0, r'{root}')"


def nb(cells):
    book = nbformat.v4.new_notebook()
    book.cells = [nbformat.v4.new_markdown_cell(c[1:]) if c.startswith("#") and not c.startswith("#!") and "\n" not in c[:2]
                  else nbformat.v4.new_code_cell(c) for c in cells]
    return book


def md(text):
    return "#" + text


BPE = [
    md("# Inspecting the learned BPE merges\n\nCFPB-trained tokenizer at V = 8k (`bpe/artifacts/cfpb_8k`). "
       "Prints tokens only, never complaint text."),
    SETUP.format(root=ROOT),
    "from bpe.tokenizer import Tokenizer\n"
    "cfpb = Tokenizer.load('bpe/artifacts/cfpb_8k')\n"
    "gen = Tokenizer.load('bpe/artifacts/general_8k')\n"
    "first = len(cfpb.specials) + 256\n"
    "print('first 30 merges (CFPB):   ', [cfpb.token_text(i) for i in range(first, first + 30)])\n"
    "print('first 30 merges (WikiText):', [gen.token_text(i) for i in range(first, first + 30)])",
    md("## Longest tokens: what earned a whole vocabulary slot"),
    "toks = [cfpb.token_text(i) for i in range(first, len(cfpb))]\n"
    "for t in sorted(toks, key=len, reverse=True)[:25]:\n"
    "    print(repr(t))",
    md("## Tokens that are artifacts of how complaints are written"),
    "for t in ['▁1681', '▁USC', '▁pursuant', '▁hereby', '▁Navient', '▁Ocwen', '▁Comenity', '▁Synchrony']:\n"
    "    print(f'{t:12s} CFPB vocab: {t in cfpb.vocab or t.replace(\"▁\", \"Ġ\") in cfpb.vocab}')\n"
    "print([t for t in toks if set(t) <= set('_*-=')][:5])",
    md("## Same sentence, two tokenizers (synthetic sentence)"),
    "s = 'I requested forbearance but the servicer reported me delinquent to Equifax under the FDCPA.'\n"
    "print('CFPB    ', len(cfpb.encode(s)), cfpb.pieces(s))\n"
    "print('WikiText', len(gen.encode(s)), gen.pieces(s))",
    md("## Redaction tokens stay atomic"),
    "s = 'On <DATE> they charged <AMT> to my account <REDACTED>.'\n"
    "print(cfpb.pieces(s))",
]

W2V = [
    md("# Probing the word2vec neighbors\n\nMy SGNS vectors (seed 0) next to gensim's, the polysemy probe, "
       "and the subword (BPE-unit) model on misspellings."),
    SETUP.format(root=ROOT),
    "from word2vec.model import Embeddings\n"
    "mine = Embeddings.load('word2vec/artifacts/seed0')\n"
    "gens = Embeddings.load('word2vec/artifacts/gensim_seed0')\n"
    "bpe = Embeddings.load('word2vec/artifacts/bpe_seed0')\n"
    "print(len(mine.words), 'words;', mine.meta['dim'], 'dims')",
    md("## Finance probes: mine vs gensim"),
    "for w in ['garnish', 'escrow', 'forbearance', 'chargeback', 'furnisher', 'autopay']:\n"
    "    print(f'{w:12s} mine:  ', [n for n, _ in mine.most_similar(w, 6)])\n"
    "    print(f'{\"\":12s} gensim:', [n for n, _ in gens.most_similar(w, 6)])",
    md("## One vector per word: polysemy"),
    "for w in ['charge', 'balance', 'dispute', 'interest']:\n"
    "    print(f'{w:9s}', [n for n, _ in mine.most_similar(w, 10)])",
    md("## Misspellings: word-level has no vector, BPE units do"),
    "for w in ['forbearnace', 'garnishmnet', 'reposession', 'delinqeunt']:\n"
    "    v = bpe.vector(w)\n"
    "    print(f'{w:13s} word-level vector: {w in mine.index}; BPE pieces: {bpe.tokenizer.pieces(\" \" + w)}')",
    md("## Analogy-style arithmetic (for curiosity; not part of the evaluation)"),
    "import numpy as np\n"
    "def analogy(a, b, c, k=5):\n"
    "    v = mine.vector(b) - mine.vector(a) + mine.vector(c)\n"
    "    sims = mine.unit @ (v / np.linalg.norm(v))\n"
    "    return [mine.words[i] for i in np.argsort(-sims) if mine.words[i] not in (a, b, c)][:k]\n"
    "print('mortgage : foreclosure :: car :', analogy('mortgage', 'foreclosure', 'car'))\n"
    "print('equifax : experian :: transunion? ->', [n for n, _ in mine.most_similar('equifax', 5)])",
]

PIPE = [
    md("# The pipeline on one synthetic complaint\n\nraw text → normalization → BPE → word2vec → seq2seq "
       "(Task B). The complaint below is written for this demo; it is not real."),
    SETUP.format(root=ROOT),
    "raw = ('On XX/XX/2021 a debt collector called me about {$1,250.00} I do not owe. '\n"
    "       'I sent a validation letter under the FDCPA to XXXX XXXX but they kept calling my work.')\n"
    "from corpus.normalize import normalize\n"
    "text = normalize(raw)\n"
    "print(text)",
    "from bpe.tokenizer import Tokenizer\n"
    "tok = Tokenizer.load('bpe/artifacts/cfpb_8k')\n"
    "ids = tok.encode(text)\n"
    "print(len(ids), 'tokens:', tok.pieces(text))",
    "from word2vec.model import Embeddings\n"
    "from word2vec.vocab import tokenize\n"
    "emb = Embeddings.load('word2vec/artifacts/seed0')\n"
    "words = [w for w in tokenize(text) if w in emb.index]\n"
    "print('in-vocabulary words:', len(words), 'of', len(tokenize(text)))\n"
    "print('neighbors of \"collector\":', [n for n, _ in emb.most_similar('collector', 5)])",
    "import json, torch\n"
    "from seq2seq.data_taskB import TaskB, truncate_word_safe\n"
    "from seq2seq.model import Seq2Seq\n"
    "from seq2seq.decode import Trie, beam_search\n"
    "from seq2seq.train import pad_batch\n"
    "run = json.loads(open('seq2seq/runs/taskB_seed0/run.json').read())\n"
    "task = TaskB(run['cfg']['tokenizer'], run['cfg']['L'])\n"
    "hp = run['hparams']\n"
    "model = Seq2Seq(hp['src_vocab'], hp['tgt_vocab'], emb=hp['emb'], hid=hp['hid'], att=hp['att'], attention=hp['attention'])\n"
    "model.load_state_dict(torch.load('seq2seq/runs/taskB_seed0/model.pt')); model.eval()\n"
    "src = truncate_word_safe(task.tok.encode(text), run['cfg']['L'], task.tok)\n"
    "x, xl = pad_batch([src])\n"
    "out = beam_search(model, x, xl, width=5, trie=Trie(task.label_ids.values()))\n"
    "print('predicted label:', task.to_label(out))",
]

BOOKS = {"bpe": (ROOT / "bpe" / "notebooks" / "inspect_merges.ipynb", BPE),
         "word2vec": (ROOT / "word2vec" / "notebooks" / "probe_neighbors.ipynb", W2V),
         "pipeline": (ROOT / "notebooks" / "00_pipeline.ipynb", PIPE)}


def build(name):
    path, cells = BOOKS[name]
    book = nb(cells)
    NotebookClient(book, timeout=900, kernel_name="python3").execute()
    # the setup cell holds a machine-specific path; blank its source in the saved copy
    book.cells[1].source = "# setup: run from the repository root (python notebooks/build.py)\n" + \
        "import sys; sys.path.insert(0, '.')"
    book.cells[1].outputs = []
    nbformat.write(book, path)
    print("wrote", path)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(BOOKS)):
        build(n)
