# Training Plan

This project should train on symbolic notation, not audio. Before model training starts, the safest path is to build a curriculum-shaped synthetic dataset, verify that every generated piece passes the validator, and only then mix in external notation datasets.

## Curriculum Ramp

| Level | Focus | Constraints |
| --- | --- | --- |
| 1 | Pulse and two notes | C-D only, no rests, no skips, quarter/half/whole notes |
| 2 | Three-note steps | C-D-E, no rests, no skips, quarter/half/whole notes |
| 3 | Five-finger steps | C-G, no rests, adjacent scale steps only |
| 4 | First skips and rests | C-G, thirds, occasional simple rests |
| 5 | One-octave C major | C4-C5, thirds, rests, no eighth notes yet |
| 6 | Neighbor keys and eighths | C/G/F, light eighth notes, controlled density |
| 7 | Early intermediate | Wider range, more keys, dotted-quarter rhythms, leaps up to a fifth |

The key idea is that each level adds one or two concepts at a time. Range expands before key signatures, key signatures arrive before dotted rhythms, and eighth notes do not appear until the student has seen an octave of simpler rhythm.

## First Training Run

Build the bootstrap dataset with the default full curriculum:

```bash
PYTHONPATH=src \
python3 training/build_synthetic_dataset.py \
  --out data/token_sequences.jsonl \
  --vocab-out data/vocab.json \
  --pieces-per-level 200
```

Then train:

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

Do not remove the validator from the generation loop after training. The model should propose candidates; the validator should still enforce the pedagogical boundaries.

## Data Mixing

After the synthetic baseline works, add real public-domain or permissively licensed notation in this order:

1. Normalize MusicXML/MIDI into the project `Piece` structure.
2. Label each piece or fragment with the strictest matching curriculum level.
3. Reject fragments that fail the matching profile.
4. Tokenize accepted fragments with the same vocabulary-building path.
5. Keep a validation split seeded and level-balanced.

External data should broaden musical variety without breaking the curriculum ramp.
