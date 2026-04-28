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

Build a bootstrap synthetic training set:

```bash
PYTHONPATH=src \
python3 training/build_synthetic_dataset.py \
  --out data/token_sequences.jsonl \
  --vocab-out data/vocab.json \
  --pieces-per-level 200
```

Train a symbolic Transformer from JSONL token-id sequences:

```bash
PYTHONPATH=src \
python3 training/train_transformer.py \
  --data data/token_sequences.jsonl \
  --epochs 20 \
  --context 256 \
  --stride 128 \
  --val-split 0.1 \
  --out models/sightreader-transformer.pt
```

The trainer slices long pieces into overlapping context windows, keeps a seeded validation split, and writes training history into the checkpoint.

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
  build_synthetic_dataset.py  Bootstrap token-id dataset builder
  train_transformer.py  PyTorch decoder-only Transformer trainer
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

## Training Data Resources

Useful symbolic-music datasets and score collections for training experiments:

- [PDMX on Zenodo](https://zenodo.org/records/15571083) - large public-domain MusicXML dataset with MXL, MIDI, metadata, validity flags, and license-conflict filters. Recommended starting point.
- [PDMX GitHub](https://github.com/pnlong/PDMX) - code and documentation for loading and working with PDMX.
- [OpenScore Lieder](https://github.com/OpenScore/Lieder) - CC0 corpus of high-quality song scores with structured metadata.
- [Mutopia Project](https://www.mutopiaproject.org/) - public-domain and Creative Commons sheet music, commonly available as LilyPond and MIDI.
- [MAESTRO](https://magenta.withgoogle.com/datasets/maestro) - piano performance MIDI/audio dataset, useful for learning pianistic motion.
- [ASAP Dataset](https://github.com/fosfrancesco/asap-dataset) - aligned piano scores and performances with MusicXML, quantized MIDI, and performance MIDI.
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) - large-scale MIDI collection; useful for scale but noisier than curated score datasets.
- [GiantMIDI-Piano](https://transactions.ismir.net/articles/10.5334/tismir.80) - large classical piano MIDI dataset described in the ISMIR Transactions dataset article.

For this project, prefer notation-first sources such as PDMX, OpenScore, and Mutopia, then filter pieces through the validator before tokenizing them for model training.
