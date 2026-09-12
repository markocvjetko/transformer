import heapq
import json
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from src.tokenizers import pretokenizers

"""
DEV NOTE: this is an actual byte-pair encoding, the v1 (BPE.py) works on characters

"""


class BytePairEncoding:
    def __init__(
        self,
        vocab_size=100,
        min_frequency=2,
        pretokenizer=pretokenizers.qwen38_pretokenizer,
        use_cache=True,
        cache_size=1_000_000
    ):

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pretokenizer = pretokenizer
        self.padding_side = "right"  # "left | right"

        self.vocab = {}  # mapping of tokens to indices
        self.inv_vocab = {}  # mapping of indices to tokens

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
            3: "<UNK>",
        }
        self.use_cache = use_cache
        self._cache_size = cache_size
        self._cache = {}

    def _merge_pair(self, tokens, pair):
        """Merge all occurrences of a token pair in the token list."""
        merged = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] is not None
                and tokens[i + 1] is not None
                and tokens[i] + tokens[i + 1] == pair
            ):
                merged.append(pair)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _tokenize_word(self, word):
        
        if self.use_cache and word in self._cache:
            return self._cache[word]

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

        if self.use_cache:
            self._cache[word] = tokens
            if len(self._cache) > self._cache_size:
                self._cache.clear()


        return tokens

    def tokenize(self, text: str, max_length: int = -1, add_special: bool = False, pad: bool = False) -> list[int]:

        if add_special and max_length > 0 and max_length < 2:
            raise ValueError(f"max_length={max_length} is too small to fit <BOS> and <EOS>")

        pretokens = self.pretokenizer(text)
        encoded = [pretokenizers.byte_encode(pretoken) for pretoken in pretokens]
        tokens = [token for b in encoded for token in self._tokenize_word(b)]

        if add_special:
            tokens = [self._special_vocab["<BOS>"]] + tokens + [self._special_vocab["<EOS>"]]
            if max_length > 0 and len(tokens) > max_length:
                tokens = tokens[: max_length - 1] + [self._special_vocab["<EOS>"]]
        elif max_length > 0:
            tokens = tokens[:max_length]

        if pad and max_length > 0 and len(tokens) < max_length:
            if self.padding_side == "right":
                tokens = tokens + [self._special_vocab["<PAD>"]] * (max_length - len(tokens))
            elif self.padding_side == "left":
                tokens = [self._special_vocab["<PAD>"]] * (max_length - len(tokens)) + tokens

        return tokens

    def decode(self, tokens: list[int] | int, add_special: bool = False) -> str:
        """
        Decodes a sequence of tokens back into the original string.

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

        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        if isinstance(tokens, int):
            tokens = [tokens]

        # If add_special == False, filter out special tokens before decoding
        if not add_special:
            tokens = [token for token in tokens if token not in self._special_inv_vocab]

        detokenized = "".join(
            [
                self._special_inv_vocab[token]
                if token in self._special_inv_vocab
                else self.inv_vocab[token]
                if token in self.inv_vocab
                else "<UNK>"
                for token in tokens
            ]
        )

        return pretokenizers.byte_decode(detokenized)

    def _build_counts(self, path: os.PathLike):

        from collections import Counter

        word_counts = Counter()
        with open(path, encoding="utf-8") as f:
            for line in f:
                word_counts.update([pretokenizers.byte_encode(pretoken) for pretoken in self.pretokenizer(line)])
        return word_counts

    def fit(self, corpus: str | os.PathLike) -> None:

        if self.vocab:
            warnings.warn("BPE fit called with non-empty vocab. Re-fitting.", stacklevel=2)
            self.vocab = {}
            self.inv_vocab = {}
            if self.use_cache:
                self._cache.clear()

        # IDs in vocab start after special vocab IDs, to avoid collisions
        idx_offset = len(self._special_inv_vocab)

        if isinstance(corpus, str):
            pretoken_counts = Counter([pretokenizers.byte_encode(pretoken) for pretoken in self.pretokenizer(corpus)])
        else:
            pretoken_counts = self._build_counts(corpus)

        for i, (_b, u) in enumerate(pretokenizers.b2u.items()):
            self.vocab[u] = i + idx_offset
            self.inv_vocab[i + idx_offset] = u

        word_tokenizations = {word: list(word) for word in pretoken_counts.keys()}

        # tracks which tokens are present for which words. For smart merging.
        token_to_words = defaultdict(set)
        token_pair_counts = defaultdict(lambda: 0)

        """
        Build a dictionary mapping a token to a list of all words where this token occurs.
        Count the number of occurences of each token pair.
        """
        for word, tokens in word_tokenizations.items():
            for token in tokens:
                token_to_words[token].add(word)
            for i in range(len(tokens) - 1):
                token_pair_counts[(tokens[i], tokens[i + 1])] += pretoken_counts[word]

        heap = []
        for pair, count in token_pair_counts.items():
            heapq.heappush(heap, (-count, pair))

        while len(self._special_vocab) + len(self.vocab) < self.vocab_size:
            # pop until we find a valid top
            while heap:
                neg_count, pair = heap[0]
                if -neg_count == token_pair_counts.get(pair, 0) and -neg_count > 0:
                    break
                heapq.heappop(heap)
            else:
                print("No token pairs found, stopping training.")
                return

            if -neg_count < self.min_frequency:
                print("No more token pairs above min_frequency, fitting stopped")
                print("Vocab size", len(self.vocab))
                return

            heapq.heappop(heap)
            token_a, token_b = pair
            most_common_pair = token_a + token_b
            next_idx = len(self.vocab)
            self.vocab[most_common_pair] = next_idx + idx_offset
            self.inv_vocab[next_idx + idx_offset] = most_common_pair

            affected = list(token_to_words[token_a] & token_to_words[token_b])
            for word in affected:
                old_tok = word_tokenizations[word]
                new_tok = self._merge_pair(old_tok, most_common_pair)
                if new_tok == old_tok:
                    continue
                word_tokenizations[word] = new_tok
                w_count = pretoken_counts[word]

                # only update pairs whose multiplicity actually changed
                old_pairs = Counter(zip(old_tok, old_tok[1:], strict=True))
                new_pairs = Counter(zip(new_tok, new_tok[1:], strict=True))
                for p in old_pairs.keys() | new_pairs.keys():
                    delta = new_pairs[p] - old_pairs[p]
                    if delta:
                        token_pair_counts[p] += delta * w_count
                        heapq.heappush(heap, (-token_pair_counts[p], p))

                if most_common_pair in new_tok:
                    token_to_words[most_common_pair].add(word)
                if token_a not in new_tok:
                    token_to_words[token_a].discard(word)
                if token_b not in new_tok:
                    token_to_words[token_b].discard(word)

            if (len(self._special_vocab) + len(self.vocab)) % 20 == 0:
                print(
                    f"Vocab size == {len(self._special_vocab) + len(self.vocab)}, \
                last merge / count == {most_common_pair, -neg_count}"
                )

    def save(self, path: str):

        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "vocab": self.vocab,
            "special_vocab": self._special_vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_file(cls, path: str, use_cache=True, cache_size=1_000_000) -> "BytePairEncoding":

        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_frequency=data["min_frequency"],
            use_cache=use_cache,
            cache_size=cache_size
        )

        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
        tokenizer._special_vocab = data["special_vocab"]
        tokenizer._special_inv_vocab = {idx: token for token, idx in tokenizer._special_vocab.items()}
        return tokenizer


if __name__ == "__main__":
    tokenizer = BytePairEncoding(vocab_size=650, min_frequency=2)
    large_paragraph = (
        "Byte Pair Encoding (BPE) is a simple form of data compression in which the most frequent pair of bytes in a sequence "
        "of bytes is replaced with a byte that does not occur within that sequence. This procedure is repeated until no more "
        "replacement is possible or a certain vocabulary size has been reached. In natural language processing, BPE is commonly "
        "used to segment words into subword units, enabling rare words to be represented as compositions of more frequent subword tokens. "
        "For example, the word 'unhappiness' could be segmented into 'un', 'happi', and 'ness', which are themselves frequent subwords. "
        "This technique helps alleviate the out-of-vocabulary problem and allows neural language models to effectively capture word "
        "compositionality. Tokenizer implementations leveraging BPE iterate over large corpora, merging the most common pairs of characters "
        "or character sequences, effectively learning the optimal set of subword units for that data. Thus, BPE is both computationally "
        "efficient and practical for building flexible and expressive tokenization schemes for a wide variety of languages. "
        "(日本語) (日本語) (日本語)"
    )
    tokenizer.fit(large_paragraph)
    tokens = tokenizer.tokenize(
        "For example, the word 'unhappiness' could be segmented into 'un', 'happi', and 'ness', which are themselves frequent subwords.",
        add_special=False,
    )
    print("decoding", tokens)
    text = tokenizer.decode(tokens, add_special=False)
    print(tokens)
    print(text)
