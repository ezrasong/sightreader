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
    duration_weights: tuple[tuple[Fraction, float], ...]
    max_interval: int
    max_scale_step: int
    max_notes_per_measure: int
    rest_probability: float
    tempo_bpm: int
    description: str

    def duration_weight(self, duration: Fraction) -> float:
        weights = dict(self.duration_weights)
        return weights.get(duration, 0.01)


PROFILES: dict[int, DifficultyProfile] = {
    1: DifficultyProfile(
        level=1,
        name="Pulse and two notes",
        allowed_keys=("C",),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("D4"),
        allowed_durations=(Fraction(1), Fraction(2), Fraction(4)),
        duration_weights=((Fraction(1), 0.72), (Fraction(2), 0.23), (Fraction(4), 0.05)),
        max_interval=2,
        max_scale_step=1,
        max_notes_per_measure=4,
        rest_probability=0.0,
        tempo_bpm=60,
        description="C-D only, steady pulse, repeats, and stepwise motion using quarter, half, and whole notes.",
    ),
    2: DifficultyProfile(
        level=2,
        name="Three-note steps",
        allowed_keys=("C",),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("E4"),
        allowed_durations=(Fraction(1), Fraction(2), Fraction(4)),
        duration_weights=((Fraction(1), 0.68), (Fraction(2), 0.26), (Fraction(4), 0.06)),
        max_interval=2,
        max_scale_step=1,
        max_notes_per_measure=4,
        rest_probability=0.0,
        tempo_bpm=64,
        description="C-D-E melodies that stay stepwise while reinforcing beat placement and note names.",
    ),
    3: DifficultyProfile(
        level=3,
        name="Five-finger steps",
        allowed_keys=("C",),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("G4"),
        allowed_durations=(Fraction(1), Fraction(2), Fraction(4)),
        duration_weights=((Fraction(1), 0.65), (Fraction(2), 0.28), (Fraction(4), 0.07)),
        max_interval=2,
        max_scale_step=1,
        max_notes_per_measure=4,
        rest_probability=0.0,
        tempo_bpm=68,
        description="C-position melodies across C-G, still limited to repeats and adjacent scale steps.",
    ),
    4: DifficultyProfile(
        level=4,
        name="First skips and rests",
        allowed_keys=("C",),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("G4"),
        allowed_durations=(Fraction(1), Fraction(2), Fraction(4)),
        duration_weights=((Fraction(1), 0.64), (Fraction(2), 0.28), (Fraction(4), 0.08)),
        max_interval=4,
        max_scale_step=2,
        max_notes_per_measure=4,
        rest_probability=0.08,
        tempo_bpm=72,
        description="C-position reading with thirds, simple phrase endings, and occasional quarter/half/whole rests.",
    ),
    5: DifficultyProfile(
        level=5,
        name="One-octave C major",
        allowed_keys=("C",),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("C5"),
        allowed_durations=(Fraction(1), Fraction(2), Fraction(4)),
        duration_weights=((Fraction(1), 0.62), (Fraction(2), 0.28), (Fraction(4), 0.10)),
        max_interval=4,
        max_scale_step=2,
        max_notes_per_measure=4,
        rest_probability=0.10,
        tempo_bpm=76,
        description="Full C-major octave with stepwise motion, thirds, and simple rests before new keys or faster notes.",
    ),
    6: DifficultyProfile(
        level=6,
        name="Neighbor keys and eighths",
        allowed_keys=("C", "G", "F"),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("C4"),
        pitch_high=pitch_name_to_midi("C5"),
        allowed_durations=(Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4)),
        duration_weights=(
            (Fraction(1, 2), 0.14),
            (Fraction(1), 0.56),
            (Fraction(2), 0.24),
            (Fraction(4), 0.06),
        ),
        max_interval=4,
        max_scale_step=2,
        max_notes_per_measure=6,
        rest_probability=0.10,
        tempo_bpm=80,
        description="C, G, and F major with light eighth-note use, thirds, and controlled event density.",
    ),
    7: DifficultyProfile(
        level=7,
        name="Early intermediate",
        allowed_keys=("C", "G", "D", "F", "Bb"),
        time_signature=(4, 4),
        pitch_low=pitch_name_to_midi("A3"),
        pitch_high=pitch_name_to_midi("E5"),
        allowed_durations=(Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(4)),
        duration_weights=(
            (Fraction(1, 2), 0.18),
            (Fraction(1), 0.48),
            (Fraction(3, 2), 0.08),
            (Fraction(2), 0.20),
            (Fraction(4), 0.06),
        ),
        max_interval=7,
        max_scale_step=4,
        max_notes_per_measure=6,
        rest_probability=0.12,
        tempo_bpm=88,
        description="Wider range, more key signatures, dotted-quarter rhythms, and leaps up to a fifth.",
    ),
}


def available_levels() -> tuple[int, ...]:
    return tuple(sorted(PROFILES))


def get_profile(level: int) -> DifficultyProfile:
    try:
        return PROFILES[level]
    except KeyError as exc:
        available = ", ".join(str(profile) for profile in sorted(PROFILES))
        raise ValueError(f"Unknown level {level}. Available levels: {available}") from exc

