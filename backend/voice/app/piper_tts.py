"""Local speech synthesis with Piper.

The default, and deliberately so. The hosted alternative needs an API key, a
model id that the provider rotates, and — as it turned out — a one-time terms
acceptance by an organisation admin before it will emit a single byte. None of
that belongs between a reviewer and a working demo of an insurance assistant.

Piper is a 63 MB ONNX voice baked into the image at build time. It runs on CPU
at roughly 5x realtime, needs no account, no network and no quota, and it makes
the voice channel work under the same "zero cost, no paid cloud" constraint as
the rest of the system. Groq synthesis remains available behind
``VOICE_TTS_PROVIDER=groq`` for anyone who prefers the better voice and has
accepted the terms.

Synthesis is CPU-bound and synchronous, so it runs in a thread rather than
blocking the event loop that is also carrying the caller's audio.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from pathlib import Path
from typing import Any

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts

log = logging.getLogger("omnicare.voice.tts")

# Piper's en_US low-quality voices are 16 kHz mono. Read from the model at load
# rather than assumed, but this is the expected value.
DEFAULT_SAMPLE_RATE = 16_000
NUM_CHANNELS = 1


class PiperTTS(tts.TTS):
    """Offline synthesis. No key, no network, no quota."""

    def __init__(self, model_path: str | Path, config_path: str | Path | None = None) -> None:
        from piper import PiperVoice

        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {model}. It is baked into the image at "
                f"build time; see the voice stage of backend/Dockerfile."
            )

        self._voice = PiperVoice.load(
            str(model),
            config_path=str(config_path) if config_path else str(model) + ".json",
        )
        sample_rate = int(
            getattr(getattr(self._voice, "config", None), "sample_rate", DEFAULT_SAMPLE_RATE)
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        log.info("Piper voice loaded from %s at %d Hz", model.name, sample_rate)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _PiperStream(tts=self, input_text=text, conn_options=conn_options)

    def render(self, text: str) -> bytes:
        """One utterance as WAV bytes. Synchronous - callers use a thread."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            self._voice.synthesize_wav(text, handle)
        return buffer.getvalue()


class _PiperStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        engine: PiperTTS = self._tts  # type: ignore[assignment]

        # to_thread because Piper is CPU-bound and synchronous; blocking here
        # would stall the same loop that is carrying the caller's audio.
        audio: bytes = await asyncio.to_thread(engine.render, self._input_text)

        output_emitter.initialize(
            request_id="piper",
            sample_rate=engine.sample_rate,
            num_channels=engine.num_channels,
            mime_type="audio/wav",
        )
        output_emitter.push(audio)
        output_emitter.flush()


def probe(model_path: str | Path, config_path: str | Path | None = None) -> tuple[bool, str]:
    """Load the voice and synthesise once, at startup.

    A voice worker that registers healthy and then cannot speak is worse than
    one that refuses to start with a reason - a lesson learned twice from the
    hosted provider, once for a missing key and once for unaccepted terms.
    """
    try:
        engine = PiperTTS(model_path, config_path)
        audio = engine.render("ok")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"{len(audio)} bytes at {engine.sample_rate} Hz"


def build(settings: Any) -> tts.TTS:
    """Select a synthesiser from configuration.

    Piper by default because it works with no account; Groq when explicitly
    asked for and its terms have been accepted.
    """
    if getattr(settings, "tts_provider", "piper") == "groq":
        from .groq_tts import GroqTTS

        return GroqTTS(
            model=settings.tts_model,
            voice=settings.tts_voice,
            base_url=settings.tts_base_url,
            api_key=settings.tts_api_key,
        )
    return PiperTTS(settings.piper_model_path, settings.piper_config_path or None)
