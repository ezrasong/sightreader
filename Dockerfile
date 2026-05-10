ARG BASE_IMAGE=pytorch/pytorch:latest

FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY training ./training
COPY tests ./tests
COPY docker/entrypoint.sh /usr/local/bin/sightreader-entrypoint

RUN python -m pip install --no-cache-dir -e . \
    && chmod +x /usr/local/bin/sightreader-entrypoint

VOLUME ["/data"]

ENV SIGHTREADER_DATA_DIR=/data
ENV PIECES_PER_LEVEL=200
ENV BARS=8
ENV EPOCHS=20
ENV BATCH_SIZE=32
ENV CONTEXT=256
ENV STRIDE=128
ENV VAL_SPLIT=0.1
ENV DEVICE=auto

ENTRYPOINT ["sightreader-entrypoint"]
CMD ["all"]
