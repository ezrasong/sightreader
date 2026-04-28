# SightReader AI

SightReader AI is a prototype for generating fresh, level-appropriate sight-reading exercises as symbolic music.

The current implementation has two layers:

- A constraint-based generator that can already create level-appropriate short pieces.
- A training scaffold for a future Transformer model that learns token sequences from symbolic music.

The generator uses structured note events as the source of truth, then exports MusicXML and MIDI.

## Quick Start

Generate an 8-bar beginner piece:

```bash
cd /Users/ezrasong/Documents/sightreader
PYTHONPATH=src \
python3 -m sightreader_ai.cli generate --level 1 --bars 8 --out outputs
```

This writes:

- `outputs/exercise.json`
- `outputs/exercise.musicxml`
- `outputs/exercise.mid`

Run tests:

```bash
PYTHONPATH=src \
python3 -m unittest discover -s tests
```

## Project Shape

```text
src/sightreader_ai/
  cli.py              Command-line entrypoint
  difficulty.py       Sight-reading difficulty profiles
  export.py           MusicXML and MIDI export
  generator.py        Constraint-based exercise generator
  music.py            Core symbolic music data structures
  tokenizer.py        Token format for neural training
  validator.py        Pedagogical constraints and scoring

training/
  train_transformer.py  Minimal PyTorch decoder-only Transformer
```

## Model Direction

The Transformer should learn tokenized symbolic music, not audio. A generated piece should pass through the validator before the app shows it to a student.

The intended pipeline is:

```text
MusicXML/MIDI dataset
  -> normalize and label difficulty
  -> tokenize
  -> train Transformer
  -> sample many candidates
  -> validate difficulty and notation
  -> export MusicXML/MIDI
```

## Documentation

Documentation lives in the [GitHub Wiki](https://github.com/ezrasong/sightreader/wiki):

- [Home](https://github.com/ezrasong/sightreader/wiki)
- [Project Overview](https://github.com/ezrasong/sightreader/wiki/Project-Overview)
- [Architecture](https://github.com/ezrasong/sightreader/wiki/Architecture)
- [Generation Pipeline](https://github.com/ezrasong/sightreader/wiki/Generation-Pipeline)
- [Training Plan](https://github.com/ezrasong/sightreader/wiki/Training-Plan)
- [Data and Licensing](https://github.com/ezrasong/sightreader/wiki/Data-and-Licensing)
- [Evaluation](https://github.com/ezrasong/sightreader/wiki/Evaluation)
- [Development](https://github.com/ezrasong/sightreader/wiki/Development)
- [Roadmap](https://github.com/ezrasong/sightreader/wiki/Roadmap)
