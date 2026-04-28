# Data and Licensing

## Data Requirements

The model needs symbolic music data:

- MIDI.
- MusicXML.
- ABC notation.
- Kern or other symbolic score formats.

For the first version, focus on short monophonic melodies or simple piano excerpts.

## Licensing Rules

Do not assume that MIDI files found online are safe for commercial use. Copyright can apply to the composition, arrangement, performance, transcription, and dataset packaging.

Preferred sources:

- Public-domain compositions.
- Explicitly permissive datasets.
- Self-created exercises.
- Procedurally generated curriculum data.

## Dataset Hygiene

Before training:

- Remove duplicates.
- Split by composition, not by small excerpt, to reduce leakage.
- Normalize keys and tempos where appropriate.
- Quantize rhythms.
- Filter invalid or overly complex files.
- Keep source and license metadata.

## Recommended Metadata

Track this for every training example:

- Source.
- License.
- Composer.
- Title.
- Key.
- Meter.
- Instrument.
- Difficulty estimate.
- Processing steps applied.

