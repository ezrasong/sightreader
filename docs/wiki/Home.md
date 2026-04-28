# SightReader AI Wiki

SightReader AI generates fresh symbolic music for sight-reading practice. The project prioritizes pedagogical control: every generated exercise should match a defined level, be readable as notation, and export cleanly to MusicXML and MIDI.

## Pages

- [Project Overview](Project-Overview.md)
- [Architecture](Architecture.md)
- [Generation Pipeline](Generation-Pipeline.md)
- [Training Plan](Training-Plan.md)
- [Data and Licensing](Data-and-Licensing.md)
- [Evaluation](Evaluation.md)
- [Development](Development.md)
- [Roadmap](Roadmap.md)

## Current Status

The current version is a functioning prototype. It includes:

- A rule-based generator for short exercises.
- Difficulty profiles for early levels.
- A validator for range, rhythm, key, intervals, and total duration.
- MusicXML, MIDI, and JSON export.
- A tokenizer for future neural training.
- A minimal PyTorch Transformer training scaffold.

The project does not yet include a trained neural model.

