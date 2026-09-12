
import regex

"""
Different ways to split sentences into pretokens.
"""

def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}

def byte_encode(text: str) -> str:
    return "".join(b2u[b] for b in text.encode("utf-8"))

def byte_decode(s: str) -> str:
    return bytes(u2b[c] for c in s).decode("utf-8", errors="replace")


gpt2_pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
qwen38_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

def _compile_pattern(pattern):
    # Use the regex module for Unicode property support (\p{L}, \p{N})
    return regex.compile(pattern)

_gpt2_regex = _compile_pattern(gpt2_pattern)
_qwen38_regex = _compile_pattern(qwen38_pattern)

def gpt2_pretokenizer(text: str):
    tokens = [token for token in _gpt2_regex.findall(text) if token]
    return tokens

def qwen38_pretokenizer(text: str):
    tokens = [token for token in _qwen38_regex.findall(text) if token]
    return tokens

def whitespace__word_tokenizer(text: str):
    return text.split()

if __name__ == "__main__":
    # # Compare how different regex pretokenizers work on several sentences


    text = "Č"    
    encoded_text = byte_encode(text)

    print(text)
    print(encoded_text)
    decoded_text = byte_decode(encoded_text)

    print(decoded_text)
