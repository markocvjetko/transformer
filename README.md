A BPE tokenizer training procedure does the following:


- splits the text into words
- tokenizes each invididual character (so that all 256 characters have a token)
- then, it looks at each word, and counts the how many adjacent token pairs there are:
    - e.g. in "low", the initial tokens are "l", "o", "w". And the pairs are "lo", "ow".
    The tokenizer does this for every word and finds the highest occuring pair, for example "he" (because of "the"
    being commonly used).


- After training, the tokenizer applies its tokenization rules sequentially (from most to least frequent), in order to process
    the whole word.

For example, suppose the tokenizer has all character level tokens, and additionally the tokens "th", "the". Then, when the word
is tokenized, the tokenizer will first split it into individual character tokens "t", "h", "e". It will then map this word into
"th", "e", and finally "the" (where "the" is represented by a single token, e.g. 1352).

A naive implementation of tokenizer training would loop over the whole corpus, each loop resulting in one newly found token.

A better imlementation could:
- count the number of occurrences for each word
- count the number of token-pairs in each word.
- this gives the information on what is the most common token pair
- this token pair is stored into the vocabulary, and all the words in the corpus are transformed accordingly
- then the process repeats


POSITIONAL EMBEDDINGS:
- SEEM TO WORK, TO CHECK IF THEY ARE EQUIVALENT TO WHAT IS IN THE OG PAPER!


31.01.2026.
positional embedding need forward pass.
Multi-head attention complete. Tested shape correctness.
embedding layer complete
Encoder block written, not tested.

BIG TODOs:

Decoder attention masking? read about it. create a new MHA for the decoder or expand the existing one?

How to handle special tokens, padding, dataloading?

Main train loop




On the Batching
- batches contained input and output sentences with aproximately the same length
- batches consisted of approximately 25k input and 25k output tokens
- base models trained for a total 100k train steps


Tokenizer: add a process batch of text function.


21.02.2026.

Transformer needs an autoregressive generation method. How can that be optimized?
Tokenizer doesn't have a beginning/end of word token (something to add).
