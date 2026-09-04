# One file, four targets. The `base` stage installs the shared dependency tree
# exactly once and Docker's layer cache reuses it across all four images -
# which is why this is a single multi-target Dockerfile rather than a separate
# base image the four services FROM. A separate base image would need building
# before `docker compose up`, and the brief promises one command.
#
#   docker compose build            builds all four, base resolved once
#   docker build --target agent .   builds one

# ---------------------------------------------------------------- base ----
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# uv rather than pip: dramatically faster resolution and install, which is what
# keeps the rebuild loop usable across four images.
RUN pip install --no-cache-dir uv

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

# Non-root. Ordinary hygiene, cheap to do, visibly absent when skipped.
RUN useradd --create-home --uid 10001 omnicare

COPY libs /app/libs

# ------------------------------------------------------------- gateway ----
FROM base AS gateway

COPY gateway /app/gateway
USER omnicare
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8080/api/v1/health')"

CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# --------------------------------------------------------------- agent ----
FROM base AS agent

COPY agent /app/agent

# The claims volume is written by this service, so it must be writable by the
# non-root user. Without this the volume mounts root-owned and submit_claim
# fails at the moment it matters most.
RUN mkdir -p /data/claims && chown -R omnicare:omnicare /data/claims

USER omnicare

# A queue consumer, not an HTTP server - no port to expose. Scale it with
# `docker compose up --scale agent=4`; the consumer group does the rest.
CMD ["python", "-m", "agent.app.worker"]

# ----------------------------------------------------------- retrieval ----
FROM base AS retrieval

COPY retrieval /app/retrieval

# Bake the embedding model into the image rather than downloading on first
# request. Otherwise the first coverage question pays for the download and the
# compose healthcheck flaps while it happens.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
 && mkdir -p /data/qdrant \
 && chown -R omnicare:omnicare /app/.fastembed /data/qdrant

USER omnicare
EXPOSE 8000

CMD ["uvicorn", "retrieval.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------- voice ----
FROM base AS voice

# Audio dependencies live only in this image. In the base they would bloat the
# gateway and agent images for nothing - a concrete payoff of keeping the voice
# worker a separate service.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir \
      "livekit-agents[silero,openai]~=1.7" \
      "piper-tts~=1.7"

# Pinned to 1.7, not the 0.12 line this was first written against. The two are
# not variations of one API: 1.x replaced the manual VAD/STT/TTS wiring with
# AgentSession, and 0.12 has no such class. Verified against the installed
# package rather than remembered.

# Bake the Piper voice in, same reasoning as the embedding model: a 63 MB
# download on first use would make the first caller wait, and would make the
# container depend on network access it otherwise does not need. Local
# synthesis is the default because the hosted alternative needs a key, a model
# id the provider rotates, and a one-time terms acceptance by an org admin.
ARG PIPER_VOICE=en_US-amy-low
ARG PIPER_URL=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low
RUN mkdir -p /app/voices \
 && curl -fsSL -o "/app/voices/${PIPER_VOICE}.onnx"      "${PIPER_URL}/${PIPER_VOICE}.onnx" \
 && curl -fsSL -o "/app/voices/${PIPER_VOICE}.onnx.json" "${PIPER_URL}/${PIPER_VOICE}.onnx.json" \
 && chown -R omnicare:omnicare /app/voices

COPY voice /app/voice
USER omnicare

# A LiveKit dispatch worker: registers with the SFU and waits to be assigned to
# rooms. No port, no HTTP surface.
CMD ["python", "-m", "voice.app.worker", "start"]
