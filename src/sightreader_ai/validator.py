from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .difficulty import DifficultyProfile
from .music import PITCH_CLASS_TO_SEMITONE, Piece, major_scale_pitches


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    messages: tuple[str, ...]
    score: float


def validate_piece(piece: Piece, profile: DifficultyProfile) -> ValidationResult:
    messages: list[str] = []

    if piece.time_signature != profile.time_signature:
        messages.append("time signature does not match profile")

    if piece.key not in profile.allowed_keys:
        messages.append(f"key {piece.key} is not allowed for level {profile.level}")

    allowed_pitches = set(major_scale_pitches(piece.key, profile.pitch_low, profile.pitch_high))
    total_duration = Fraction(0)
    last_pitch: int | None = None
    leaps = 0

    for note in piece.notes:
        total_duration += note.duration
        if note.pitch < profile.pitch_low or note.pitch > profile.pitch_high:
            messages.append(f"{note.pitch_name} is outside allowed range")
        if note.pitch not in allowed_pitches:
            messages.append(f"{note.pitch_name} is outside {piece.key} major")
        if note.duration not in profile.allowed_durations:
            messages.append(f"duration {note.duration} is not allowed")
        if last_pitch is not None:
            interval = abs(note.pitch - last_pitch)
            if interval > profile.max_interval:
                messages.append(f"interval of {interval} semitones is too large")
            if interval >= 5:
                leaps += 1
        last_pitch = note.pitch

    if total_duration != piece.total_beats():
        messages.append(f"duration total {total_duration} does not fill {piece.total_beats()} beats")

    if piece.notes and piece.notes[-1].pitch % 12 != PITCH_CLASS_TO_SEMITONE[piece.key]:
        messages.append("piece does not end on the tonic pitch class")

    density = len(piece.notes) / max(piece.measures, 1)
    leap_penalty = min(leaps / max(len(piece.notes), 1), 1.0)
    density_penalty = max(0.0, density - (4.5 if profile.level == 1 else 6.5)) * 0.08
    score = max(0.0, 1.0 - leap_penalty * 0.35 - density_penalty)

    return ValidationResult(ok=not messages, messages=tuple(messages), score=score)
