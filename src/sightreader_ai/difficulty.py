from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .music import pitch_name_to_midi


@dataclass(frozen=True)
class DifficultyProfile:
    level: int
    name: str
    allowed_keys: tuple[str, ...]
    time_signature: tuple[int, int]
    pitch_low: int
    pitch_high: int
    allowed_durations: tuple[Fraction, ...]
    max_interval: int
    tempo_bpm: int
    description: str


PROFILES: dict[int, DifficultyProfile] = {
    1: DifficultyProfile(
        level=1,
        name="First position",
        allowed_keys=("C",),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("G4"),
        allowed_durations=(Fraction(1), Fraction(2), Fraction(4)),
        max_interval=4,
        tempo_bpm=72,
        description="Stepwise C-major melodies using quarter, half, and whole notes.",
    ),
    2: DifficultyProfile(
        level=2,
        name="Expanding range",
        allowed_keys=("C", "G", "F"),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("C5"),
        allowed_durations=(Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4)),
        max_interval=7,
        tempo_bpm=84,
        description="Simple major keys with eighth notes and small leaps.",
    ),
    3: DifficultyProfile(
        level=3,
        name="Early intermediate",
        allowed_keys=("C", "G", "D", "F", "Bb"),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("A3"),
        pitch_high=pitch_name_to_midi("E5"),
        allowed_durations=(Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(4)),
        max_interval=12,
        tempo_bpm=96,
        description="Wider range, dotted rhythms, and more key signatures.",
    ),
}


def get_profile(level: int) -> DifficultyProfile:
    try:
        return PROFILES[level]
    except KeyError as exc:
        available = ", ".join(str(profile) for profile in sorted(PROFILES))
        raise ValueError(f"Unknown level {level}. Available levels: {available}") from exc

