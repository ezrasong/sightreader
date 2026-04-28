# Training Plan

## Goal

Train a symbolic music model that can generate notation-like token sequences under sight-reading constraints.

The first trained model should generate monophonic or simple piano exercises. It should not generate raw audio.

## Token Format

The tokenizer emits control tokens and note tokens:

```text
LEVEL_1 KEY_C METER_4_4 BARS_8
BAR POS_0 PITCH_C4 DUR_1
POS_1 PITCH_D4 DUR_1
EOS
```

Control tokens tell the model what kind of exercise to generate.

## Training Data Shape

Training examples should be JSONL records:

```json
{"ids": [1, 14, 20, 45, 91, 92, 93]}
```

The vocabulary should be saved alongside the model so generated token IDs can be decoded later.

## Initial Model

Use the scaffold in `training/train_transformer.py`:

- Decoder-only Transformer.
- Next-token prediction.
- Cross-entropy loss.
- Padding token ignored by the loss.

## Recommended Training Stages

1. Train on generated curriculum data to test the pipeline.
2. Add public-domain symbolic music.
3. Label examples by difficulty.
4. Train a small model from scratch.
5. Add constrained sampling.
6. Compare model-generated exercises against the rule-based generator.

## Success Criteria

A useful model should:

- Produce valid token sequences.
- Decode into valid notes.
- Match requested level controls.
- Produce variety beyond the rule-based generator.
- Pass validation at a high rate.

