from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sightreader_ai.difficulty import get_profile
from sightreader_ai.export import export_piece
from sightreader_ai.generator import ExerciseGenerator, GenerationOptions
from sightreader_ai.tokenizer import SightReadingTokenizer
from sightreader_ai.validator import validate_piece
from training.train_transformer import _make_examples, _split_examples


class GeneratorTests(unittest.TestCase):
    def test_generates_valid_level_one_piece(self) -> None:
        piece = ExerciseGenerator(seed=12).generate(GenerationOptions(level=1, bars=8, seed=12))
        result = validate_piece(piece, get_profile(1))

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(piece.measures, 8)
        self.assertEqual(piece.time_signature, (4, 4))

    def test_generates_valid_level_two_pieces_across_seeds(self) -> None:
        for seed in range(10):
            piece = ExerciseGenerator(seed=seed).generate(GenerationOptions(level=2, bars=4, seed=seed))
            result = validate_piece(piece, get_profile(2))
            self.assertTrue(result.ok, (seed, piece.key, result.messages))

    def test_exports_musicxml_and_midi(self) -> None:
        piece = ExerciseGenerator(seed=7).generate(GenerationOptions(level=2, bars=4, seed=7))

        with TemporaryDirectory() as directory:
            paths = export_piece(piece, Path(directory))
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["musicxml"].read_text(encoding="utf-8").startswith("<?xml"))
            self.assertEqual(paths["midi"].read_bytes()[:4], b"MThd")

    def test_tokenizer_adds_control_tokens(self) -> None:
        piece = ExerciseGenerator(seed=3).generate(GenerationOptions(level=1, bars=2, seed=3))
        tokens = SightReadingTokenizer().encode(piece)

        self.assertEqual(tokens[0], "LEVEL_1")
        self.assertIn("KEY_C", tokens)
        self.assertEqual(tokens[-1], "EOS")

    def test_training_examples_use_overlapping_windows(self) -> None:
        examples = _make_examples([[1, 2, 3, 4, 5, 6, 7]], context=3, stride=2)

        self.assertEqual(examples, [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7]])

    def test_training_split_keeps_validation_when_available(self) -> None:
        examples = [[1, 2], [2, 3], [3, 4]]
        train_examples, val_examples = _split_examples(examples, val_split=0.1, seed=4)

        self.assertEqual(len(train_examples), 2)
        self.assertEqual(len(val_examples), 1)


if __name__ == "__main__":
    unittest.main()
