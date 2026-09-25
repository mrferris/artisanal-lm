"What I cannot create, I do not understand" - Richard Feynman

Implemented by hand, from scratch, with love, for learning purposes:
 - BPE tokenizer training & encoding
 - Llama-3 era transformer model implementation (RoPE, RMSNorm, SwiGLU)
 - Distributed DP pre-training
 - GRPO & DPO post-training
 - Performance benchmarking

To train:
```bash
python -m lm.train_model --help
```
