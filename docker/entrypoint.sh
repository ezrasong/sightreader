#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${SIGHTREADER_DATA_DIR:-/data}"
DATASET="${DATASET:-${DATA_DIR}/token_sequences.jsonl}"
VOCAB="${VOCAB:-${DATA_DIR}/vocab.json}"
MODEL_OUT="${MODEL_OUT:-${DATA_DIR}/models/sightreader-transformer.pt}"

PIECES_PER_LEVEL="${PIECES_PER_LEVEL:-200}"
BARS="${BARS:-8}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONTEXT="${CONTEXT:-256}"
STRIDE="${STRIDE:-128}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-1337}"
LEVELS="${LEVELS:-}"

mkdir -p "${DATA_DIR}" "$(dirname "${MODEL_OUT}")"

level_args=()
if [[ -n "${LEVELS}" ]]; then
  read -r -a level_values <<< "${LEVELS}"
  level_args=(--levels "${level_values[@]}")
fi

build_dataset() {
  python training/build_synthetic_dataset.py \
    --out "${DATASET}" \
    --vocab-out "${VOCAB}" \
    --pieces-per-level "${PIECES_PER_LEVEL}" \
    --bars "${BARS}" \
    --seed "${SEED}" \
    "${level_args[@]}"
}

train_model() {
  if [[ ! -s "${DATASET}" ]]; then
    echo "Dataset not found at ${DATASET}; building it first."
    build_dataset
  fi

  python training/train_transformer.py \
    --data "${DATASET}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --context "${CONTEXT}" \
    --stride "${STRIDE}" \
    --val-split "${VAL_SPLIT}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --out "${MODEL_OUT}"
}

case "${1:-all}" in
  all)
    build_dataset
    train_model
    ;;
  build-dataset)
    build_dataset
    ;;
  train)
    train_model
    ;;
  test)
    python -m unittest discover -s tests
    ;;
  shell)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
