import argparse
from collections import Counter
from bpe_tokenizer.tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a Byte-Pair Encoding (BPE) Tokenizer.")
    parser.add_argument("--corpus", required=True, help="Path to training corpus text file.")
    parser.add_argument("--merges", type=int, default=1000, help="Number of BPE merges to perform.")
    parser.add_argument("--out", required=True, help="Path to output JSON file for saved merges.")
    args = parser.parse_args()

    # Read corpus file
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    print(f"Training tokenizer on '{args.corpus}' with {args.merges} merges...")
    tokenizer = Tokenizer.from_corpus(corpus_text, args.merges)
    
    # Save trained merges
    tokenizer.save(args.out)
    print(f"Tokenizer successfully trained and saved to '{args.out}'.")

if __name__ == "__main__":
    main()
