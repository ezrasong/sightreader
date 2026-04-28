from __future__ import annotations

import json
import struct
from fractions import Fraction
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from .music import Piece, midi_to_musicxml_pitch


DIVISIONS = 4
TICKS_PER_QUARTER = 480


def write_json(piece: Piece, path: Path) -> None:
    path.write_text(json.dumps(piece.to_json(), indent=2) + "\n", encoding="utf-8")


def write_musicxml(piece: Piece, path: Path) -> None:
    root = Element("score-partwise", version="4.0")
    part_list = SubElement(root, "part-list")
    score_part = SubElement(part_list, "score-part", id="P1")
    SubElement(score_part, "part-name").text = "Piano"

    part = SubElement(root, "part", id="P1")
    notes_by_measure = _group_notes_by_measure(piece)

    for measure_number in range(1, piece.measures + 1):
        measure = SubElement(part, "measure", number=str(measure_number))
        if measure_number == 1:
            attributes = SubElement(measure, "attributes")
            SubElement(attributes, "divisions").text = str(DIVISIONS)
            key = SubElement(attributes, "key")
            SubElement(key, "fifths").text = str(_key_fifths(piece.key))
            time = SubElement(attributes, "time")
            SubElement(time, "beats").text = str(piece.time_signature[0])
            SubElement(time, "beat-type").text = str(piece.time_signature[1])
            clef = SubElement(attributes, "clef")
            SubElement(clef, "sign").text = "G"
            SubElement(clef, "line").text = "2"
            direction = SubElement(measure, "direction", placement="above")
            sound = SubElement(direction, "sound")
            sound.set("tempo", str(piece.tempo_bpm))

        for note_event in notes_by_measure.get(measure_number, []):
            note = SubElement(measure, "note")
            pitch = SubElement(note, "pitch")
            step, alter, octave = midi_to_musicxml_pitch(note_event.pitch)
            SubElement(pitch, "step").text = step
            if alter is not None:
                SubElement(pitch, "alter").text = str(alter)
            SubElement(pitch, "octave").text = str(octave)
            SubElement(note, "duration").text = str(_duration_to_divisions(note_event.duration))
            SubElement(note, "type").text = _duration_type(note_event.duration)

    pretty_xml = minidom.parseString(tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    path.write_text(pretty_xml, encoding="utf-8")


def write_midi(piece: Piece, path: Path) -> None:
    events: list[tuple[int, bytes]] = []
    events.append((0, _meta_tempo(piece.tempo_bpm)))
    events.append((0, bytes([0xC0, 0])))  # Acoustic grand piano.

    current_tick = 0
    for note in piece.notes:
        duration_ticks = _beats_to_ticks(note.duration)
        events.append((current_tick, bytes([0x90, note.pitch, note.velocity])))
        events.append((current_tick + duration_ticks, bytes([0x80, note.pitch, 0])))
        current_tick += duration_ticks
    events.append((current_tick, b"\xff\x2f\x00"))

    track_data = bytearray()
    last_tick = 0
    for tick, payload in sorted(events, key=lambda item: (item[0], item[1][0])):
        track_data.extend(_variable_length_quantity(tick - last_tick))
        track_data.extend(payload)
        last_tick = tick

    header = b"MThd" + struct.pack(">LHHH", 6, 0, 1, TICKS_PER_QUARTER)
    track = b"MTrk" + struct.pack(">L", len(track_data)) + bytes(track_data)
    path.write_bytes(header + track)


def export_piece(piece: Piece, output_dir: Path, basename: str = "exercise") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{basename}.json",
        "musicxml": output_dir / f"{basename}.musicxml",
        "midi": output_dir / f"{basename}.mid",
    }
    write_json(piece, paths["json"])
    write_musicxml(piece, paths["musicxml"])
    write_midi(piece, paths["midi"])
    return paths


def _group_notes_by_measure(piece: Piece) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for note in piece.notes:
        grouped.setdefault(note.measure, []).append(note)
    return grouped


def _duration_to_divisions(duration: Fraction) -> int:
    return int(duration * DIVISIONS)


def _beats_to_ticks(duration: Fraction) -> int:
    return int(duration * TICKS_PER_QUARTER)


def _duration_type(duration: Fraction) -> str:
    return {
        Fraction(4): "whole",
        Fraction(3): "half",
        Fraction(2): "half",
        Fraction(3, 2): "quarter",
        Fraction(1): "quarter",
        Fraction(1, 2): "eighth",
    }.get(duration, "quarter")


def _key_fifths(key: str) -> int:
    return {
        "Cb": -7,
        "Gb": -6,
        "Db": -5,
        "Ab": -4,
        "Eb": -3,
        "Bb": -2,
        "F": -1,
        "C": 0,
        "G": 1,
        "D": 2,
        "A": 3,
        "E": 4,
        "B": 5,
        "F#": 6,
        "C#": 7,
    }[key]


def _meta_tempo(bpm: int) -> bytes:
    microseconds_per_quarter = int(60_000_000 / bpm)
    return b"\xff\x51\x03" + microseconds_per_quarter.to_bytes(3, "big")


def _variable_length_quantity(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    bytes_out = [buffer]
    while value:
        buffer = (value & 0x7F) | 0x80
        bytes_out.insert(0, buffer)
        value >>= 7
    return bytes(bytes_out)

