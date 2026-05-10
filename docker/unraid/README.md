# Unraid Template Notes

The Unraid template lives at:

```text
templates/sightreader-ai-trainer.xml
```

The repository also includes `ca_profile.xml`, `LICENSE`, and an icon under `icons/`, matching the shape expected by Community Applications style template repositories.

The template runs SightReader AI as a job container. It builds the synthetic curriculum dataset, trains the symbolic Transformer, writes outputs to a persistent data path, then exits.

## Manual Install

Copy the template to:

```text
/boot/config/plugins/dockerMan/templates-user/
```

Then install it from Docker > Add Container > User Templates.

## Image

The template uses:

```text
ghcr.io/ezrasong/sightreader-ai-trainer:latest
```

When a new image is published from `main`, use Unraid's Docker update flow to pull it. The mapped `/data` path keeps generated datasets and model checkpoints outside the image.

## Persistent Paths

Default host path:

```text
/mnt/cache/appdata/sightreader
```

Container path:

```text
/data
```

Expected outputs:

```text
/data/token_sequences.jsonl
/data/vocab.json
/data/models/sightreader-transformer.pt
```

## Job Modes

The template defaults to `all`, which rebuilds the dataset and trains the model.

Set Post Arguments to one of these values when you want a narrower job:

```text
build-dataset
train
test
shell
```

`train` will automatically build the dataset first if `/data/token_sequences.jsonl` is missing.

## GPU Notes

For NVIDIA GPU training, install the Unraid Nvidia-Driver plugin, set Extra Parameters to:

```text
--runtime=nvidia
```

and keep:

```text
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
DEVICE=auto
```

For CPU-only training, leave Extra Parameters blank, leave `NVIDIA_VISIBLE_DEVICES` blank, and set:

```text
DEVICE=cpu
```
