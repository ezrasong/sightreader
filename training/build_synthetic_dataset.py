from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sightreader_ai.generator import ExerciseGenerator, GenerationOptions
from sightreader_ai.tokenizer import SightReadingTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bootstrap JSONL training set from generated sight-reading exercises."
    )
    parser.add_argument("--out", type=Path, default=Path("data/token_sequences.jsonl"))
    parser.add_argument("--vocab-out", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--pieces-per-level", type=int, default=200)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    build_dataset(
        out=args.out,
        vocab_out=args.vocab_out,
        levels=args.levels,
        pieces_per_level=args.pieces_per_level,
        bars=args.bars,
        seed=args.seed,
    )


def build_dataset(
    *,
    out: Path,
    vocab_out: Path,
    levels: list[int],
    pieces_per_level: int,
    bars: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    tokenizer = SightReadingTokenizer()
    rows: list[dict[str, object]] = []
    token_sequences: list[list[str]] = []

    for level in levels:
        generator = ExerciseGenerator(seed=rng.randrange(1_000_000_000))
        for index in range(pieces_per_level):
            piece_seed = rng.randrange(1_000_000_000)
            piece = generator.generate(
                GenerationOptions(
                    level=level,
                    bars=bars,
                    seed=piece_seed,
                    title=f"Synthetic L{level} #{index + 1}",
                )
            )
            tokens = ["BOS", *tokenizer.encode(piece)]
            token_sequences.append(tokens)
            rows.append({"level": level, "seed": piece_seed, "tokens": tokens})

    vocab = tokenizer.build_vocab(token_sequences)
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab_out.parent.mkdir(parents=True, exist_ok=True)
    vocab_out.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            tokens = row["tokens"]
            assert isinstance(tokens, list)
            handle.write(
                json.dumps(
                    {
                        "ids": tokenizer.ids(tokens, vocab),
                        "level": row["level"],
                        "seed": row["seed"],
                    }
                )
                + "\n"
            )

    print(f"wrote {len(rows)} sequences to {out}")
    print(f"wrote {len(vocab)} tokens to {vocab_out}")


if __name__ == "__main__":
    main()
