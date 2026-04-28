# Project Overview

## Goal

SightReader AI helps musicians practice sight-reading by generating short pieces they have never seen before. The app should produce notation that is playable, level-appropriate, and varied enough to build real reading skill.

## Core Product Loop

1. The learner chooses a level, instrument, length, and optional style.
2. The system generates a new exercise.
3. The exercise is validated against sight-reading constraints.
4. The learner reads the notation and optionally plays back the MIDI.
5. Future versions score performance and adapt the next exercise.

## Why Symbolic Music

The project uses symbolic music instead of raw audio because sight-reading depends on notation. The source of truth should be structured notes, measures, durations, key signatures, and metadata.

Primary formats:

- JSON for internal structured data.
- MusicXML for notation rendering and score interchange.
- MIDI for playback and download.

## Design Principle

The model should generate candidates. The validator should decide what is appropriate for practice.

