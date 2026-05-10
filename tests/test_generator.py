from __future__ import annotations

import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from sightreader_ai.difficulty import available_levels, get_profile
from sightreader_ai.export import export_piece
from sightreader_ai.generator import ExerciseGenerator, GenerationOptions
from sightreader_ai.music import Piece, pitch_name_to_midi
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

    def test_all_curriculum_levels_generate_valid_pieces(self) -> None:
        for level in available_levels():
            for seed in range(3):
                piece = ExerciseGenerator(seed=seed).generate(GenerationOptions(level=level, bars=4, seed=seed))
                result = validate_piece(piece, get_profile(level))
                self.assertTrue(result.ok, (level, seed, piece.key, result.messages))

    def test_curriculum_introduces_fundamentals_gradually(self) -> None:
        levels = [get_profile(level) for level in available_levels()]

        self.assertEqual([profile.level for profile in levels], list(range(1, 8)))
        self.assertLess(levels[0].pitch_high, levels[1].pitch_high)
        self.assertLess(levels[1].pitch_high, levels[2].pitch_high)
        self.assertEqual(levels[0].max_scale_step, 1)
        self.assertEqual(levels[1].max_scale_step, 1)
        self.assertEqual(levels[2].max_scale_step, 1)
        self.assertNotIn(Fraction(1, 2), levels[4].allowed_durations)
        self.assertIn(Fraction(1, 2), levels[5].allowed_durations)
        self.assertEqual(levels[2].rest_probability, 0.0)
        self.assertGreater(levels[3].rest_probability, 0.0)

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

    def test_rest_events_tokenize_and_export(self) -> None:
        piece = Piece.from_notes(
            title="Rest basics",
            key="C",
            time_signature=(4, 4),
            tempo_bpm=72,
            level=4,
            measures=1,
            pitches_and_durations=[
                (pitch_name_to_midi("C4"), Fraction(1)),
                (None, Fraction(1)),
                (pitch_name_to_midi("D4"), Fraction(1)),
                (pitch_name_to_midi("C4"), Fraction(1)),
            ],
        )
        result = validate_piece(piece, get_profile(4))
        tokens = SightReadingTokenizer().encode(piece)

        self.assertTrue(result.ok, result.messages)
        self.assertIn("REST", tokens)
        with TemporaryDirectory() as directory:
            paths = export_piece(piece, Path(directory))
            self.assertIn("<rest", paths["musicxml"].read_text(encoding="utf-8"))
            self.assertEqual(paths["midi"].read_bytes()[:4], b"MThd")

    def test_early_levels_reject_rests(self) -> None:
        piece = Piece.from_notes(
            title="Too early for rests",
            key="C",
            time_signature=(4, 4),
            tempo_bpm=60,
            level=1,
            measures=1,
            pitches_and_durations=[
                (pitch_name_to_midi("C4"), Fraction(1)),
                (None, Fraction(1)),
                (pitch_name_to_midi("D4"), Fraction(1)),
                (pitch_name_to_midi("C4"), Fraction(1)),
            ],
        )
        result = validate_piece(piece, get_profile(1))

        self.assertFalse(result.ok)
        self.assertIn("rests are not allowed for level 1", result.messages)

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
