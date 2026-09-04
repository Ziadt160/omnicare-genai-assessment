"""Voice worker settings.

STT and TTS both point at OpenAI-compatible endpoints, so Groq's free Whisper
serves transcription with the same client that would talk to OpenAI - the same
observation that collapses four LLM providers into one class.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICE_", extra="ignore")

    livekit_url: str = "ws://livekit:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "devsecret-local-only-not-for-production"

    redis_url: str = "redis://redis:6379/0"

    # Free tier. Verify current model ids when wiring - Groq rotates them.
    stt_base_url: str = "https://api.groq.com/openai/v1"
    stt_model: str = "whisper-large-v3-turbo"
    stt_api_key: str = ""

    # "piper" is the default: local, 63 MB baked into the image, no key and no
    # quota. "groq" sounds better but needs an API key and a one-time terms
    # acceptance by an org admin before it returns a single byte - not a thing
    # to put between a reviewer and a working demo.
    tts_provider: Literal["piper", "groq"] = "piper"

    piper_model_path: str = "/app/voices/en_US-amy-low.onnx"
    piper_config_path: str = ""

    tts_base_url: str = "https://api.groq.com/openai/v1"
    tts_model: str = "canopylabs/orpheus-v1-english"
    tts_voice: str = "tara"
    tts_api_key: str = ""

    # Much tighter than the REST budget: dead air reads as a dropped call long
    # before a slow answer reads as thinking.
    # One real synthesis request at boot. Config presence is not correctness,
    # and both TTS failures so far were only visible from an actual call.
    verify_speech_on_start: bool = True

    run_timeout_s: float = 6.0
    low_confidence_threshold: float = 0.6
