# Architecture

## Current Architecture

```text
CLI
  -> ExerciseGenerator
  -> DifficultyProfile
  -> Validator
  -> Exporter
      -> JSON
      -> MusicXML
      -> MIDI
```

## Source Modules

- `src/sightreader_ai/music.py`: core symbolic music data structures.
- `src/sightreader_ai/difficulty.py`: level definitions and constraints.
- `src/sightreader_ai/generator.py`: constraint-based music generation.
- `src/sightreader_ai/validator.py`: exercise validation and scoring.
- `src/sightreader_ai/export.py`: JSON, MusicXML, and MIDI export.
- `src/sightreader_ai/tokenizer.py`: token format for neural training.
- `src/sightreader_ai/cli.py`: command-line interface.
- `training/train_transformer.py`: minimal decoder-only Transformer scaffold.

## Intended App Architecture

```text
Frontend
  React or SwiftUI
  score rendering
  playback controls

Backend
  FastAPI
  generation endpoint
  model sampling endpoint
  exercise validation

Model Layer
  symbolic tokenizer
  Transformer model
  constrained sampler

Storage
  generated exercises
  user progress
  practice history
```

## Important Boundary

Do not let the neural model be the only source of correctness. The validator should remain a separate deterministic layer so the app can reject invalid or inappropriate generations.

