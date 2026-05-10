from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

from .difficulty import DifficultyProfile, get_profile
from .music import PITCH_CLASS_TO_SEMITONE, Piece, major_scale_pitches
from .validator import ValidationResult, validate_piece


@dataclass(frozen=True)
class GenerationOptions:
    level: int = 1
    bars: int = 8
    seed: int | None = None
    title: str = "Sight Reading Exercise"


class ExerciseGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def generate(self, options: GenerationOptions) -> Piece:
        profile = get_profile(options.level)
        rng = random.Random(options.seed) if options.seed is not None else self._random

        for _ in range(200):
            key = rng.choice(profile.allowed_keys)
            piece = self._generate_candidate(profile, key, options.bars, options.title, rng)
            result = validate_piece(piece, profile)
            if result.ok:
                return piece

        messages = "; ".join(validate_piece(piece, profile).messages)
        raise RuntimeError(f"Could not generate a valid exercise: {messages}")

    def _generate_candidate(
        self,
        profile: DifficultyProfile,
        key: str,
        bars: int,
        title: str,
        rng: random.Random,
    ) -> Piece:
        scale = major_scale_pitches(key, profile.pitch_low, profile.pitch_high)
        tonic_pitch_class = PITCH_CLASS_TO_SEMITONE[key]
        tonic_candidates = [pitch for pitch in scale if pitch % 12 == tonic_pitch_class]
        tonic = min(tonic_candidates, key=lambda pitch: abs(pitch - scale[len(scale) // 2]))

        durations = self._make_rhythm(profile, bars, rng)
        pitches = self._make_melody(
            scale=scale,
            tonic=tonic,
            count=len(durations),
            max_interval=profile.max_interval,
            max_scale_step=profile.max_scale_step,
            rng=rng,
        )
        pitches = self._apply_rests(pitches, profile.rest_probability, rng)
        return Piece.from_notes(
            title=title,
            key=key,
            time_signature=profile.time_signature,
            tempo_bpm=profile.tempo_bpm,
            level=profile.level,
            measures=bars,
            pitches_and_durations=zip(pitches, durations),
        )

    def _make_rhythm(
        self,
        profile: DifficultyProfile,
        bars: int,
        rng: random.Random,
    ) -> list[Fraction]:
        beats_per_measure = Fraction(profile.time_signature[0] * 4, profile.time_signature[1])
        durations: list[Fraction] = []
        allowed = sorted(profile.allowed_durations)

        for measure_index in range(bars):
            remaining = beats_per_measure
            while remaining > 0:
                candidates = [duration for duration in allowed if duration <= remaining]
                if measure_index == bars - 1 and remaining in candidates:
                    duration = remaining
                else:
                    weights = [profile.duration_weight(duration) for duration in candidates]
                    duration = rng.choices(candidates, weights=weights, k=1)[0]
                durations.append(duration)
                remaining -= duration
        return durations

    def _make_melody(
        self,
        *,
        scale: list[int],
        tonic: int,
        count: int,
        max_interval: int,
        max_scale_step: int,
        rng: random.Random,
    ) -> list[int]:
        current = tonic
        pitches: list[int] = []
        weighted_steps, weights = self._scale_step_weights(max_scale_step)

        for index in range(count):
            if index == count - 1:
                current = tonic
            elif index > 0 and index % 8 == 7:
                current = self._nearest_chord_tone(scale, tonic, current)
            else:
                scale_index = scale.index(current)
                for _ in range(20):
                    step = rng.choices(weighted_steps, weights=weights, k=1)[0]
                    next_index = min(max(scale_index + step, 0), len(scale) - 1)
                    candidate = scale[next_index]
                    if abs(candidate - current) <= max_interval:
                        current = candidate
                        break
            pitches.append(current)
        return pitches

    @staticmethod
    def _scale_step_weights(max_scale_step: int) -> tuple[list[int], list[float]]:
        steps = [0]
        weights = [0.22]
        for distance in range(1, max_scale_step + 1):
            weight = 0.36 / distance
            steps.extend((distance, -distance))
            weights.extend((weight, weight))
        return steps, weights

    @staticmethod
    def _apply_rests(
        pitches: list[int],
        rest_probability: float,
        rng: random.Random,
    ) -> list[int | None]:
        if rest_probability <= 0:
            return list(pitches)

        last_index = len(pitches) - 1
        output: list[int | None] = []
        for index, pitch in enumerate(pitches):
            if index in (0, last_index):
                output.append(pitch)
            elif rng.random() < rest_probability:
                output.append(None)
            else:
                output.append(pitch)
        return output

    @staticmethod
    def _nearest_chord_tone(scale: list[int], tonic: int, current: int) -> int:
        chord_classes = {(tonic + interval) % 12 for interval in (0, 4, 7)}
        chord_tones = [pitch for pitch in scale if pitch % 12 in chord_classes]
        return min(chord_tones, key=lambda pitch: abs(pitch - current))


def generate_validated(options: GenerationOptions) -> tuple[Piece, ValidationResult]:
    generator = ExerciseGenerator(seed=options.seed)
    piece = generator.generate(options)
    return piece, validate_piece(piece, get_profile(options.level))
