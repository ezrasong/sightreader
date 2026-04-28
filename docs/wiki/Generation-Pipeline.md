# Generation Pipeline

## Current Pipeline

1. Load a difficulty profile.
2. Choose an allowed key.
3. Generate rhythm that fills each measure.
4. Generate melody from the major scale.
5. Prefer stepwise motion and small leaps.
6. End on the tonic.
7. Validate the piece.
8. Export JSON, MusicXML, and MIDI.

## Difficulty Constraints

Each profile defines:

- Allowed keys.
- Time signature.
- Pitch range.
- Allowed durations.
- Maximum melodic interval.
- Tempo.
- Description of the target reading level.

## Validation Checks

The validator currently checks:

- Time signature.
- Key.
- Pitch range.
- Diatonic pitch membership.
- Allowed durations.
- Maximum interval.
- Total duration.
- Tonic ending.

## Why Rejection Sampling

The generator can try multiple candidates and keep the first valid result. This is simple and effective for early levels. Later, neural generation should use the same strategy: sample several candidates and keep the best validated exercise.

