from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable


PITCH_CLASS_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

SEMITONE_TO_SHARP_NAME = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

MAJOR_SCALE_STEPS = (0, 2, 4, 5, 7, 9, 11)


def pitch_name_to_midi(name: str) -> int:
    pitch = name[:-1]
    octave = int(name[-1])
    if pitch not in PITCH_CLASS_TO_SEMITONE:
        raise ValueError(f"Unknown pitch name: {name}")
    return 12 * (octave + 1) + PITCH_CLASS_TO_SEMITONE[pitch]


def midi_to_pitch_name(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{SEMITONE_TO_SHARP_NAME[midi % 12]}{octave}"


def midi_to_musicxml_pitch(midi: int) -> tuple[str, int | None, int]:
    name = SEMITONE_TO_SHARP_NAME[midi % 12]
    octave = midi // 12 - 1
    if len(name) == 2:
        return name[0], 1, octave
    return name, None, octave


def major_scale_pitches(key: str, low: int, high: int) -> list[int]:
    tonic = PITCH_CLASS_TO_SEMITONE[key]
    return [
        midi
        for midi in range(low, high + 1)
        if (midi - tonic) % 12 in MAJOR_SCALE_STEPS
    ]


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    duration: Fraction
    measure: int
    beat: Fraction
    velocity: int = 72

    @property
    def pitch_name(self) -> str:
        return midi_to_pitch_name(self.pitch)

    def to_json(self) -> dict[str, object]:
        return {
            "pitch": self.pitch_name,
            "midi": self.pitch,
            "duration": str(self.duration),
            "measure": self.measure,
            "beat": str(self.beat),
            "velocity": self.velocity,
        }


@dataclass(frozen=True)
class Piece:
    title: str
    key: str
    time_signature: tuple[int, int]
    tempo_bpm: int
    level: int
    measures: int
    notes: tuple[NoteEvent, ...] = field(default_factory=tuple)

    @property
    def beats_per_measure(self) -> Fraction:
        numerator, denominator = self.time_signature
        return Fraction(numerator * 4, denominator)

    def total_beats(self) -> Fraction:
        return self.beats_per_measure * self.measures

    def to_json(self) -> dict[str, object]:
        return {
            "title": self.title,
            "key": self.key,
            "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}",
            "tempo_bpm": self.tempo_bpm,
            "level": self.level,
            "measures": self.measures,
            "notes": [note.to_json() for note in self.notes],
        }

    @classmethod
    def from_notes(
        cls,
        *,
        title: str,
        key: str,
        time_signature: tuple[int, int],
        tempo_bpm: int,
        level: int,
        measures: int,
        pitches_and_durations: Iterable[tuple[int, Fraction]],
    ) -> "Piece":
        beats_per_measure = Fraction(time_signature[0] * 4, time_signature[1])
        notes: list[NoteEvent] = []
        cursor = Fraction(0)
        for pitch, duration in pitches_and_durations:
            measure = int(cursor // beats_per_measure) + 1
            beat = cursor % beats_per_measure
            notes.append(NoteEvent(pitch=pitch, duration=duration, measure=measure, beat=beat))
            cursor += duration
        return cls(
            title=title,
            key=key,
            time_signature=time_signature,
            tempo_bpm=tempo_bpm,
            level=level,
            measures=measures,
            notes=tuple(notes),
        )

