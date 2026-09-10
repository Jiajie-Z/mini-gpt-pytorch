"""Train a small character-level GPT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import CharTokenizer, get_batch, split_data
from model import GPTLanguageModel


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    eval_iters: int,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(data)

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.block_size,
                args.batch_size,
                args.device,
                args.eval_iters,
            )
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(
            "train", train_data, val_data, args.block_size, args.batch_size, args.device
        )
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gpt_char_model.pt")
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer.chars), encoding="utf-8")
    print(f"saved model and tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
