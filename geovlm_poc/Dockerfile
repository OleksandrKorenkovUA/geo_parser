FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ARG PIP_EXTRAS=""
RUN if [ -n "$PIP_EXTRAS" ]; then \
        pip install --no-cache-dir .[${PIP_EXTRAS}]; \
    else \
        pip install --no-cache-dir .; \
    fi

ENTRYPOINT ["geovlm-poc"]
