# Roadmap

## Phase 1: Generator Foundation

- Create structured music model.
- Generate level-based exercises.
- Export JSON, MusicXML, and MIDI.
- Add tests for generation and export.

Status: started.

## Phase 2: Dataset Pipeline

- Import MIDI and MusicXML.
- Normalize rhythm and key.
- Extract short exercises.
- Label difficulty features.
- Save tokenized JSONL files.

## Phase 3: First Neural Model

- Build vocabulary.
- Train small Transformer from scratch.
- Save checkpoints.
- Add sampling script.
- Decode generated tokens back to structured music.

## Phase 4: Constrained AI Generation

- Generate multiple candidates from the model.
- Validate each candidate.
- Rank by readability and variety.
- Export the best candidate.

## Phase 5: Practice App

- Add notation UI.
- Add playback controls.
- Add user settings.
- Track practice history.
- Adapt future exercises based on performance.

## Phase 6: Performance Feedback

- Add MIDI keyboard input or microphone input.
- Compare performed notes and rhythm against the exercise.
- Score accuracy.
- Recommend targeted follow-up exercises.

