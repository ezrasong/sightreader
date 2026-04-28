# Development

## Local Setup

Run commands from the repository root:

```bash
cd /Users/ezrasong/Documents/sightreader
```

The current generator has no required third-party dependencies.

## Generate an Exercise

```bash
PYTHONPATH=src python3 -m sightreader_ai.cli generate --level 1 --bars 8 --out outputs
```

Use a seed for repeatable output:

```bash
PYTHONPATH=src python3 -m sightreader_ai.cli generate --level 2 --bars 8 --seed 9 --out outputs
```

## Run Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Compile Check

```bash
PYTHONPATH=src python3 -m compileall src training tests
```

## Install Training Dependencies

The training script expects PyTorch:

```bash
python3 -m pip install -e '.[train]'
```

## Repository Hygiene

Generated outputs, local datasets, model checkpoints, virtual environments, and Python caches are ignored by `.gitignore`.

