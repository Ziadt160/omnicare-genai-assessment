"""Retrieval settings."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    source_dir: str = "/data/source"
    qdrant_path: str = "/data/qdrant"
    # "memory" is the default and is adequate for a two-section corpus; the
    # Qdrant adapter exists so the port has a real second implementation and
    # so a larger policy library is one env var away.
    vector_backend: Literal["memory", "qdrant"] = "memory"
    default_top_k: int = 3
