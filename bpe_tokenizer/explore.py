import matplotlib.pyplot as plt
from bpe_tokenizer.tokenizer import Tokenizer

def run_exploration():
    corpus = """
    Byte-pair encoding learns a vocabulary of subword units from a corpus. 
    Frequent words stay whole, rare words split into reusable pieces, so the model never meets a truly unknown word.
    Training starts from characters and repeatedly merges the most frequent adjacent pair into a new symbol.
    Encoding a new word splits it into characters, then applies learned merges in order.
    """
    
    # Test different merge counts
    merge_counts = [0, 5, 10, 20, 50, 100, 200]
    avg_tokens_per_word = []
    
    words = corpus.strip().split()
    num_words = len(words)

    for m in merge_counts:
        tokenizer = Tokenizer.from_corpus(corpus, num_merges=m)
        total_tokens = sum(len(tokenizer.encode(word)) for word in words)
        avg_tokens = total_tokens / num_words
        avg_tokens_per_word.append(avg_tokens)
        print(f"Merges: {m:3d} | Avg Tokens/Word: {avg_tokens:.2f}")

    # Plotting vocabulary size / merges vs. average tokens per word
    plt.figure(figsize=(8, 5))
    plt.plot(merge_counts, avg_tokens_per_word, marker='o', linestyle='-', color='b')
    plt.title("BPE Merges vs. Average Tokens Per Word")
    plt.xlabel("Number of Merges (Vocabulary Growth)")
    plt.ylabel("Average Tokens per Word")
    plt.grid(True)
    plt.savefig("bpe_exploration.png")
    print("\nExploration plot saved successfully as 'bpe_exploration.png'.")

if __name__ == "__main__":
    run_exploration()
