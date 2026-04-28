# Evaluation

## Generator Evaluation

The generator should be evaluated on:

- Validity rate.
- Difficulty match.
- Rhythmic clarity.
- Melodic readability.
- Variety across seeds.
- Export correctness.

## Model Evaluation

The trained model should be evaluated on:

- Token validity.
- Decode success rate.
- Validator pass rate.
- Requested level match.
- Novelty compared with training data.
- Human playability.

## Automated Checks

Current tests verify:

- Level 1 generation.
- Level 2 generation across multiple seeds.
- MusicXML and MIDI export.
- Tokenizer control tokens.

## Human Review

Human review is still required. A piece can pass mechanical validation and still feel awkward or musically weak.

Useful review questions:

- Is the notation readable at the intended level?
- Does the rhythm feel natural?
- Are leaps prepared or too surprising?
- Does the phrase have a clear ending?
- Would a teacher assign this?

