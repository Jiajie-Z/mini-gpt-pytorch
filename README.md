# Mini GPT PyTorch

A step-by-step PyTorch implementation of a small GPT-style decoder-only Transformer for character-level language modeling.

This project started as a learning notebook and was refactored into a reusable repo with modular model code, training, generation, and documentation.

## Features

- Character-level tokenizer for TinyShakespeare-style text
- Token and positional embeddings
- Causal self-attention with a lower-triangular mask
- Multi-head attention
- Feed-forward network and residual Transformer blocks
- Cross-entropy next-token prediction loss
- Autoregressive text generation
- CPU/GPU-compatible training scripts

## Project Structure

```text
mini-gpt-pytorch/
├── README.md
├── requirements.txt
├── data/
│   └── input.txt
├── notebooks/
│   └── GPT-Step-by-Step.ipynb
├── outputs/
│   └── sample_generation.txt
└── src/
    ├── dataset.py
    ├── generate.py
    ├── model.py
    └── train.py
```

## Model Architecture

```text
Input token ids
-> Token Embedding + Positional Embedding
-> Transformer Blocks
   -> LayerNorm
   -> Causal Multi-Head Self-Attention
   -> Residual Connection
   -> LayerNorm
   -> Feed Forward Network
   -> Residual Connection
-> Final LayerNorm
-> Linear Language Modeling Head
-> Next-token logits
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a small model:

```bash
python src/train.py --max-iters 2000
```

Generate text:

```bash
python src/generate.py --prompt "I tell you,"
```

For a quick CPU smoke test:

```bash
python src/train.py --max-iters 2 --eval-interval 1 --eval-iters 1 --batch-size 4 --block-size 8 --n-embd 16 --n-head 4 --n-layer 1
```

## Dataset

The project is designed for the TinyShakespeare character-level dataset. Put the dataset at:

```text
data/input.txt
```

You can download it with:

```bash
curl -fsSL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt
```

## What I Learned

- How GPT predicts the next token from prior context
- Why decoder-only models need causal masking
- How Q/K/V attention and multi-head attention work
- How cross-entropy loss is used for language modeling
- How autoregressive text generation repeatedly feeds predictions back into the model

## Resume Summary

Implemented a GPT-style decoder-only Transformer from scratch in PyTorch, including token embeddings, positional embeddings, causal self-attention, multi-head attention, feed-forward layers, training loop, evaluation, and autoregressive text generation.
