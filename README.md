31.01.2026.
positional embedding need forward pass.
Multi-head attention complete. Tested shape correctness.
embedding layer complete
Encoder block written, not tested.



On Batching:
- "In Attention is All You Need" batches contained input and output sentences with aproximately the same length
- batches consisted of approximately 25k input and 25k output tokens
- base model trained for a total 100k train steps

21.02.2026.

Transformer needs an autoregressive generation method. How can that be optimized?
Tokenizer doesn't have a beginning/end of word token (something to add).
