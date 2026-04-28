from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrainConfig:
    data: Path
    epochs: int
    batch_size: int
    context: int
    stride: int
    d_model: int
    layers: int
    heads: int
    lr: float
    weight_decay: float
    dropout: float
    grad_clip: float
    val_split: float
    seed: int
    device: str
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small symbolic-music Transformer.")
    parser.add_argument("--data", type=Path, required=True, help="JSONL file of token id sequences")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Token stride for slicing long pieces into overlapping training examples.",
    )
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", type=Path, default=Path("models/sightreader-transformer.pt"))
    args = parser.parse_args()
    train(
        TrainConfig(
            data=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            context=args.context,
            stride=args.stride,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            grad_clip=args.grad_clip,
            val_split=args.val_split,
            seed=args.seed,
            device=args.device,
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
    examples = _make_examples(sequences, context=config.context, stride=config.stride)
    train_examples, val_examples = _split_examples(examples, val_split=config.val_split, seed=config.seed)
    device = _resolve_device(config.device, torch)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    class TokenDataset(Dataset):
        def __init__(self, rows: Sequence[list[int]]) -> None:
            self._rows = rows

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, index: int):
            sequence = self._rows[index]
            padded = sequence + [0] * max(0, config.context + 1 - len(sequence))
            tensor = torch.tensor(padded, dtype=torch.long)
            return tensor[:-1], tensor[1:]

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        context=config.context,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loader_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TokenDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            TokenDataset(val_examples),
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        if val_examples
        else None
    )

    print(
        "training "
        f"device={device} sequences={len(sequences)} train_examples={len(train_examples)} "
        f"val_examples={len(val_examples)} vocab_size={vocab_size}"
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        train_loss = _run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            vocab_size=vocab_size,
            device=device,
            grad_clip=config.grad_clip,
            torch=torch,
        )
        metrics: dict[str, float | int] = {"epoch": epoch + 1, "train_loss": train_loss}
        message = f"epoch={epoch + 1} train_loss={train_loss:.4f}"
        if val_loader is not None:
            val_loss = _evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                vocab_size=vocab_size,
                device=device,
                torch=torch,
            )
            metrics["val_loss"] = val_loss
            message += f" val_loss={val_loss:.4f}"
        history.append(metrics)
        print(message)

    config.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": _checkpoint_config(config),
            "vocab_size": vocab_size,
            "history": history,
        },
        config.out,
    )


def _load_sequences(path: Path) -> list[list[int]]:
    sequences: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        sequence = value["ids"] if isinstance(value, dict) else value
        if len(sequence) >= 2:
            sequences.append(sequence)
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    return sequences


def _make_examples(sequences: Sequence[Sequence[int]], *, context: int, stride: int) -> list[list[int]]:
    if context < 1:
        raise ValueError("context must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    window = context + 1
    examples: list[list[int]] = []
    for sequence in sequences:
        if len(sequence) <= window:
            examples.append(list(sequence))
            continue
        for start in range(0, len(sequence) - window + 1, stride):
            examples.append(list(sequence[start : start + window]))
        if examples[-1] != list(sequence[-window:]):
            examples.append(list(sequence[-window:]))
    return examples


def _split_examples(
    examples: Sequence[list[int]],
    *,
    val_split: float,
    seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * val_split)
    if val_split > 0 and len(shuffled) > 1:
        val_count = max(1, val_count)
    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:] or val_examples
    return train_examples, val_examples


def _resolve_device(device_name: str, torch):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(device_name)


def _run_train_epoch(
    *,
    model,
    loader,
    optimizer,
    loss_fn,
    vocab_size: int,
    device,
    grad_clip: float,
    torch,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def _evaluate(*, model, loader, loss_fn, vocab_size: int, device, torch) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def _checkpoint_config(config: TrainConfig) -> dict[str, object]:
    values = dict(config.__dict__)
    values["data"] = str(config.data)
    values["out"] = str(config.out)
    return values


class DecoderOnlyTransformer:
    def __new__(cls, *args, **kwargs):
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                context: int,
                d_model: int,
                layers: int,
                heads: int,
                dropout: float,
            ) -> None:
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(context, d_model)
                self.dropout = nn.Dropout(dropout)
                block = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(block, num_layers=layers)
                self.norm = nn.LayerNorm(d_model)
                self.output = nn.Linear(d_model, vocab_size)
                self.context = context

            def forward(self, inputs):
                batch, length = inputs.shape
                positions = torch.arange(length, device=inputs.device).unsqueeze(0).expand(batch, length)
                hidden = self.token_embedding(inputs) + self.position_embedding(positions)
                hidden = self.dropout(hidden)
                mask = torch.triu(torch.ones(length, length, device=inputs.device), diagonal=1).bool()
                hidden = self.transformer(hidden, mask=mask)
                hidden = self.norm(hidden)
                return self.output(hidden)

        return _Model(*args, **kwargs)


if __name__ == "__main__":
    main()

