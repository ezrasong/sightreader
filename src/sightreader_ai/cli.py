from __future__ import annotations

import argparse
from pathlib import Path

from .export import export_piece
from .generator import ExerciseGenerator, GenerationOptions
from .tokenizer import SightReadingTokenizer
from .validator import validate_piece
from .difficulty import get_profile


def main() -> None:
    parser = argparse.ArgumentParser(prog="sightreader-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a sight-reading exercise")
    generate.add_argument("--level", type=int, default=1)
    generate.add_argument("--bars", type=int, default=8)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--title", default="Sight Reading Exercise")
    generate.add_argument("--out", type=Path, default=Path("outputs"))
    generate.add_argument("--basename", default="exercise")
    generate.add_argument("--print-tokens", action="store_true")

    args = parser.parse_args()

    if args.command == "generate":
        options = GenerationOptions(
            level=args.level,
            bars=args.bars,
            seed=args.seed,
            title=args.title,
        )
        piece = ExerciseGenerator(seed=args.seed).generate(options)
        result = validate_piece(piece, get_profile(args.level))
        paths = export_piece(piece, args.out, args.basename)

        print(f"Generated: {piece.title}")
        print(f"Validation score: {result.score:.2f}")
        for kind, path in paths.items():
            print(f"{kind}: {path}")

        if args.print_tokens:
            print("tokens:")
            print(" ".join(SightReadingTokenizer().encode(piece)))


if __name__ == "__main__":
    main()

