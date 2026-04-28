from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data: Path
    epochs: int
    batch_size: int
    context: int
    d_model: int
    layers: int
    heads: int
    lr: float
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small symbolic-music Transformer.")
    parser.add_argument("--data", type=Path, required=True, help="JSONL file of token id sequences")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=Path("models/sightreader-transformer.pt"))
    args = parser.parse_args()
    train(
        TrainConfig(
            data=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            context=args.context,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            lr=args.lr,
            out=args.out,
        )
    )


def train(config: TrainConfig) -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: pip install -e '.[train]'") from exc

    sequences = _load_sequences(config.data)
    vocab_size = max(max(sequence) for sequence in sequences) + 1

    class TokenDataset(Dataset):
        def __len__(self) -> int:
            return len(sequences)

        def __getitem__(self, index: int):
            sequence = sequences[index][: config.context + 1]
            padded = sequence + [0] * max(0, config.context + 1 - len(sequence))
            tensor = torch.tensor(padded, dtype=torch.long)
            return tensor[:-1], tensor[1:]

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        context=config.context,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loader = DataLoader(TokenDataset(), batch_size=config.batch_size, shuffle=True)

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(len(loader), 1):.4f}")

    config.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.__dict__, "vocab_size": vocab_size}, config.out)


def _load_sequences(path: Path) -> list[list[int]]:
    sequences: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        sequences.append(value["ids"] if isinstance(value, dict) else value)
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    return sequences


class DecoderOnlyTransformer:
    def __new__(cls, *args, **kwargs):
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(self, vocab_size: int, context: int, d_model: int, layers: int, heads: int) -> None:
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(context, d_model)
                block = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                    activation="gelu",
                )
                self.transformer = nn.TransformerEncoder(block, num_layers=layers)
                self.output = nn.Linear(d_model, vocab_size)
                self.context = context

            def forward(self, inputs):
                batch, length = inputs.shape
                positions = torch.arange(length, device=inputs.device).unsqueeze(0).expand(batch, length)
                hidden = self.token_embedding(inputs) + self.position_embedding(positions)
                mask = torch.triu(torch.ones(length, length, device=inputs.device), diagonal=1).bool()
                hidden = self.transformer(hidden, mask=mask)
                return self.output(hidden)

        return _Model(*args, **kwargs)


if __name__ == "__main__":
    main()

