import os
import urllib.request

def download_corpus():
    os.makedirs("data", exist_ok=True)
    corpus_path = "data/corpus.txt"
    
    # Using Arthur Conan Doyle's 'The Adventures of Sherlock Holmes' from Project Gutenberg
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    
    print("Downloading corpus from Project Gutenberg...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        text = response.read().decode('utf-8')
        
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"Corpus successfully saved to '{corpus_path}' ({len(text)} characters).")

if __name__ == "__main__":
    download_corpus()
