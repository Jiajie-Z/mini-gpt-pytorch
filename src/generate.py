"""Generate text from a trained character-level GPT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import CharTokenizer
from model import GPTLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/gpt_char_model.pt")
    parser.add_argument("--tokenizer", default="outputs/tokenizer.json")
    parser.add_argument("--prompt", default="I tell you,")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chars = json.loads(Path(args.tokenizer).read_text(encoding="utf-8"))
    tokenizer = CharTokenizer(chars=chars)

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    generated = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
