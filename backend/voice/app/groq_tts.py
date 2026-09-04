"""Groq speech synthesis, without the field Groq rejects.

`livekit-plugins-openai`'s TTS is *almost* right for Groq - same endpoint, same
body - but it always sends `stream_format`, on the documented assumption that
OpenAI-compatible servers ignore unknown fields. Groq does not ignore it:

    400 - unknown field `stream_format` in request body

so every synthesis fails and the caller hears nothing. Sixty lines here beat
monkey-patching a third-party plugin's request body, and the failure mode this
avoids is the worst kind: the session connects, the agent answers, the browser
shows the transcript, and the line is silent.

Groq's TTS models require the organisation to accept the provider's terms once
at https://console.groq.com - until then every request returns 400 with a
message saying so, which `check_config` surfaces at startup rather than leaving
to be discovered mid-call.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    tts,
)

log = logging.getLogger("omnicare.voice.tts")

SAMPLE_RATE = 24_000
NUM_CHANNELS = 1


class GroqTTS(tts.TTS):
    """Non-streaming synthesis against an OpenAI-shaped `/audio/speech`.

    Non-streaming on purpose: Groq returns the whole clip, and a policyholder
    answer is a sentence or two. Chasing byte-level streaming here would add a
    failure mode to save a few hundred milliseconds on an utterance that has
    already waited on retrieval and a model.
    """

    def __init__(
        self,
        *,
        model: str,
        voice: str,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        response_format: str = "wav",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._model = model
        self._voice = voice
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._response_format = response_format

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _GroqStream(tts=self, input_text=text, conn_options=conn_options)


class _GroqStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        engine: GroqTTS = self._tts  # type: ignore[assignment]

        body: dict[str, Any] = {
            "model": engine._model,
            "voice": engine._voice,
            "input": self._input_text,
            "response_format": engine._response_format,
            # Deliberately no `stream_format`. That single field is the whole
            # reason this class exists.
        }

        async with httpx.AsyncClient(timeout=self._conn_options.timeout) as client:
            response = await client.post(
                f"{engine._base_url}/audio/speech",
                headers={"Authorization": f"Bearer {engine._api_key}"},
                json=body,
            )

            if response.status_code != 200:
                detail = response.text[:300]
                if "terms acceptance" in detail:
                    # Actionable, and not something retrying will ever fix.
                    log.error(
                        "Groq TTS model %s needs one-time terms acceptance at "
                        "https://console.groq.com - synthesis will fail until then",
                        engine._model,
                    )
                raise APIStatusError(
                    message=detail,
                    status_code=response.status_code,
                    retryable=response.status_code >= 500,
                )

            output_emitter.initialize(
                request_id=response.headers.get("x-request-id", "groq-tts"),
                sample_rate=engine.sample_rate,
                num_channels=engine.num_channels,
                mime_type=f"audio/{engine._response_format}",
            )
            output_emitter.push(response.content)
            output_emitter.flush()


def probe(model: str, voice: str, api_key: str,
          base_url: str = "https://api.groq.com/openai/v1") -> tuple[bool, str]:
    """One synchronous request, so misconfiguration is found at startup.

    Returns (ok, detail). Called by `check_config` - a voice worker that
    registers healthy and then cannot speak is worse than one that refuses to
    start with a reason.
    """
    if not api_key:
        return False, "no API key"
    try:
        r = httpx.post(
            f"{base_url.rstrip('/')}/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "voice": voice, "input": "ok",
                  "response_format": "wav"},
            timeout=30,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if r.status_code == 200:
        return True, f"{len(r.content)} bytes"
    return False, r.text[:200]
