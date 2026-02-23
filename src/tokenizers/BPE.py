from collections import Counter, defaultdict
import re
import json

from pathlib import Path

class BytePairEncoding: #Byte Pair Encoding

    def __init__(self, vocab_size=100, min_frequency=2):
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
    
        self.vocab = {} #mapping of tokens to indices
        self.inv_vocab = {} #mapping of indices to tokens

        self._special_vocab = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }
        self._special_inv_vocab = {
            0: "<PAD>",
            1: "<BOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

    def _merge_pair(self, tokens, pair):
        """Merge all occurrences of a token pair in the token list."""
        merged = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 
                and tokens[i] is not None 
                and tokens[i + 1] is not None 
                and tokens[i] + tokens[i + 1] == pair):
                merged.append(pair)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _tokenize_word(self, word):

        tokens = [ch if ch in self.vocab else None for ch in word]

        while len(tokens) > 1:
            # Only consider pairs where neither token is None (unknown)
            token_pairs = [
                tokens[i] + tokens[i + 1]
                for i in range(len(tokens) - 1)
                if tokens[i] is not None and tokens[i + 1] is not None
            ]

            valid_token_pairs = [pair for pair in token_pairs if pair in self.vocab]
            if not valid_token_pairs:
                break

            most_common = min(valid_token_pairs, key=lambda x: self.vocab[x])
            tokens = self._merge_pair(tokens, most_common)

        tokens = [self.vocab[token] if token is not None else self._special_vocab["<UNK>"] for token in tokens]

        return tokens

    def tokenize(
        self, 
        text: str, 
        max_length: int = -1, 
        add_special: bool = False, 
        pad: bool = False
    ) -> list[int]:

        if add_special and max_length > 0 and max_length < 2:
            raise ValueError(f"max_length={max_length} is too small to fit <BOS> and <EOS>")

        words = re.findall(r'\S+|\s+', text)  # matches non-whitespace OR whitespace
        tokens = [token for word in words for token in self._tokenize_word(word)]

        if add_special:
            tokens = [self._special_vocab["<BOS>"]] + tokens + [self._special_vocab["<EOS>"]]
            if max_length > 0 and len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self._special_vocab["<EOS>"]]
        elif max_length > 0:
            tokens = tokens[:max_length]
        
        if pad and max_length > 0 and len(tokens) < max_length:
            tokens += [self._special_vocab["<PAD>"]] * (max_length - len(tokens)) 
        return tokens

    def decode(self, tokens: list[int] | int, add_special: bool = False) -> str:
        """
        Decodes a sequence of token(s) back into the original string.

        Args:
            tokens (list[int] | int): A list of token indices (ints), 
                                        or a single token (int) to decode.
            add_special (bool, optional): Whether to include special tokens (such as <BOS>, <EOS>, or <PAD>) in the decoded string.
                                            Defaults to False.

        Returns:
            str: The decoded string corresponding to the input tokens.
        """
        if tokens is None:
            return ""
        
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()

        if isinstance(tokens, int):
            tokens = [tokens]

        # Fix: avoid crash if tokens is None or not iterable
        # If add_special is False, filter out special tokens (<BOS>, <EOS>, <PAD>) before decoding

        if not add_special:
            tokens = [token for token in tokens if token not in self._special_inv_vocab]

        return "".join(
            self._special_inv_vocab[token] if token in self._special_inv_vocab
            else self.inv_vocab[token] if token in self.inv_vocab
            else "<UNK>"
            for token in tokens
        )

    def decode_batch():
        # TODO how should this be done (should decode itself accept batches)
        # An extra function seems unncessary
        raise NotImplementedError

    def fit(self, corpus: str) -> None:
        # If the vocabulary is not empty, do not fit again.

        if self.vocab:
            print("BPE fit called with non-empty vocab. Re-fitting.")
            self.vocab = {}
            self.inv_vocab = {}

        #IDs in vocab start after IDs in special vocab, to avoid collisions
        #with special tokens
        idx_offset = len(self._special_inv_vocab)
        
        words = corpus.split()
        counts = Counter(words)

        unique_chars = sorted(set(corpus))
        for i, char in enumerate(unique_chars):
            self.vocab[char] = i + idx_offset
            self.inv_vocab[i + idx_offset] = char

        word_tokenizations = {word: list(word) for word in words}

        while len(self._special_vocab) + len(self.vocab) < self.vocab_size:
            #count all token pairs
            token_pair_counts = defaultdict(int)
            for word in counts:
                tokenized_word = word_tokenizations[word]
                for i in range(len(tokenized_word) - 1):
                    substring = "".join(tokenized_word[i:i+2])
                    token_pair_counts[substring] += counts[word]

            if not token_pair_counts:
                print("No token pairs found, stopping training.")
                return

            most_common_pair = max(token_pair_counts, key=token_pair_counts.get)

            if token_pair_counts[most_common_pair] < self.min_frequency:
                    print("No more token pairs above min_frequency, fitting stopped")
                    print("Vocab size", len(self.vocab))
                    return

            #add the most common_token to the tokenizer structs
            next_idx = len(self.vocab)
            self.vocab[most_common_pair] = next_idx + idx_offset
            self.inv_vocab[next_idx + idx_offset] = most_common_pair

            # apply to tokenized_words (modifies tokenized_words in place)
            for word, tokens in word_tokenizations.items():
                word_tokenizations[word] = self._merge_pair(tokens, most_common_pair)

    def save(self, path: str):

        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "vocab": self.vocab,
            "special_vocab": self._special_vocab
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_file(cls, path: str) -> "BytePairEncoding":

        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_frequency=data["min_frequency"]
        )

        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
        tokenizer._special_vocab = data["special_vocab"]
        tokenizer._special_inv_vocab = {idx: token for token, idx in tokenizer._special_vocab.items()}
        return tokenizer

    def preprocess(self, text):
        """
        Removes leading and trailing whitespaces.
        Removes consecutive whitespaces.
        WARNING! Removes tabs, new lines, etc.
        """
        return " ".join(text.split())


    
if __name__ == "__main__":

    tokenizer = BytePairEncoding(vocab_size=600, min_frequency=2)
    large_paragraph = (
        "Byte Pair Encoding (BPE) is a simple form of data compression in which the most frequent pair of bytes in a sequence "
        "of bytes is replaced with a byte that does not occur within that sequence. This procedure is repeated until no more "
        "replacement is possible or a certain vocabulary size has been reached. In natural language processing, BPE is commonly "
        "used to segment words into subword units, enabling rare words to be represented as compositions of more frequent subword tokens. "
        "For example, the word 'unhappiness' could be segmented into 'un', 'happi', and 'ness', which are themselves frequent subwords. "
        "This technique helps alleviate the out-of-vocabulary problem and allows neural language models to effectively capture word "
        "compositionality. Tokenizer implementations leveraging BPE iterate over large corpora, merging the most common pairs of characters "
        "or character sequences, effectively learning the optimal set of subword units for that data. Thus, BPE is both computationally "
        "efficient and practical for building flexible and expressive tokenization schemes for a wide variety of languages."
    )
    tokenizer.fit(large_paragraph)
    #print(tokenizer.vocab)
    tokens = tokenizer.tokenize("practical", add_special=True)
    print("decoding", tokens)
    text = tokenizer.decode(tokens, add_special=True)
    print(tokens)
    print(text)


    from datasets import load_dataset

    # Load a small subset from wikitext-2 (e.g., first 1000 samples)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:1000]")
    wiki_texts = "\n".join(ds["text"])
    tokenizer.fit(wiki_texts)
    #print(tokenizer.vocab)
    print(tokenizer.tokenize("This sentence will be tokenized"))
    print([tokenizer.decode([token]) for token in tokenizer.tokenize("This sentece will be tokenized")])
    print(tokenizer.decode(tokenizer.tokenize("This sentence will be tokenized")))
    print(tokenizer.decode(tokenizer.tokenize("token")))
    print(tokenizer.tokenize("token"))
    print([tokenizer.decode([token], add_special=True) for token in tokenizer.tokenize("token", add_special=True)])
    
    # Correct - specify a file name
    tokenizer.save("./config/tokenizer.json")

    bpe_loaded = BytePairEncoding.from_file("./config/tokenizer.json")

    # Verify they are the same
    assert tokenizer.vocab_size == bpe_loaded.vocab_size, "Vocab sizes do not match"
    assert tokenizer.min_frequency == bpe_loaded.min_frequency, "Min frequencies do not match"
    assert tokenizer.vocab == bpe_loaded.vocab, "Vocabs do not match"
    assert tokenizer.inv_vocab == bpe_loaded.inv_vocab, "Inverse vocabs do not match"
    print("Tokenizer and loaded BPE are the same!")
