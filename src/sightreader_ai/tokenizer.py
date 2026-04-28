from __future__ import annotations

from fractions import Fraction

from .music import Piece, midi_to_pitch_name


class SightReadingTokenizer:
    """Small text-token format for future neural training."""

    def encode(self, piece: Piece) -> list[str]:
        tokens = [
            f"LEVEL_{piece.level}",
            f"KEY_{piece.key}",
            f"METER_{piece.time_signature[0]}_{piece.time_signature[1]}",
            f"BARS_{piece.measures}",
        ]
        current_measure = 0
        for note in piece.notes:
            if note.measure != current_measure:
                tokens.append("BAR")
                current_measure = note.measure
            tokens.extend(
                [
                    f"POS_{_fraction_token(note.beat)}",
                    f"PITCH_{midi_to_pitch_name(note.pitch)}",
                    f"DUR_{_fraction_token(note.duration)}",
                ]
            )
        tokens.append("EOS")
        return tokens

    def build_vocab(self, token_sequences: list[list[str]]) -> dict[str, int]:
        vocab = {"PAD": 0, "BOS": 1, "UNK": 2}
        for sequence in token_sequences:
            for token in sequence:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    def ids(self, tokens: list[str], vocab: dict[str, int]) -> list[int]:
        return [vocab.get(token, vocab["UNK"]) for token in tokens]


def _fraction_token(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}_{value.denominator}"

