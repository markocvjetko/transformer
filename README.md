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
- Done, check github issues
Tokenizer doesn't have a beginning/end of word token (something to add).
- Done, check github issues

05.03.2026.
- Considered doing a refactor, decided not to. It seems to much work compared to the immediate gains. The current codebase is easy to maintain and augment. Once it gets bigger a larger refactor will be consider.
- what eventually needs to be refactored. How the datasets, models, trainers and tokenizers interact with each other. E.g. currently the model class accepts tokenizers in the constructor, does this make sense?

The training for multi30k seems to work, reaching around BLEU of 18.7 and still improving. What to do with it? Ideas:
- implement decoder only:
    - gpt-2 paper https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- hparam optim for optimizers (Adam, Muon, lr...)
- hparam optim for architecture?
- more fun: Train a model with no spaces. How to tokenize?
- more fun: train with/without tokenizer
    - https://arxiv.org/html/2508.13666v1 (training on code with/without formatting)
    - scriptio continua https://en.wikipedia.org/wiki/Scriptio_continua#:~:text=%E4%BD%8D%E9%BA%BB%E9%A0%88-,Thai%20script,endings%20of%20clauses%20or%20sentences.


07.03.2026.

Added beam search toy implementation in beam_search_toy.ipynb
- The implementation keeps finished beams inside the beam list, stops after all beams are finished
- no per-batch optimization in the toy model (e.g. batch elements with all beams finished 
are still fed to the transformer)



13.03.2026.

Considered a hyperparameter optim library
- Optuna previously used, seems to work well
- Ray Tune - seems too heavy for what I need. It seeems to be a hparam optim for very large scale experiments. 
For the purposes of this codebase, Optuna syncing via a journal.log file works well across different HPC (jean zay) nodes.

Refactored lightning training config managment

Todays commits should have been on main, not beam_search branch