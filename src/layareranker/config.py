"""Runtime settings, read from `LAYARERANKER_*` environment variables or passed explicitly."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LongDocPolicy = Literal["truncate", "window_max"]


class Settings(BaseSettings):
    """Every knob that changes scores or throughput. Frozen so a loaded scorer cannot drift."""

    model_config = SettingsConfigDict(env_prefix="LAYARERANKER_", frozen=True, extra="forbid")

    # Model and scoring behaviour (all of these feed `score_version`).
    model: str = "english"
    revision: str | None = None
    device: str | None = None
    preset: str = "default"
    long_doc: LongDocPolicy = "truncate"
    window_stride_ratio: float = Field(0.5, gt=0.0, le=1.0)
    max_windows_per_doc: int = Field(8, ge=1)
    max_query_tokens: int = Field(96, ge=8)
    min_passage_tokens: int = Field(96, ge=16)

    # Forward-pass sizing. Rows are (passage, question) sequences; tokens are rows x padded length.
    max_batch_rows: int = Field(64, ge=1)
    max_batch_tokens: int = Field(32768, ge=512)

    # Micro-batcher.
    max_wait_ms: float = Field(5.0, ge=0.0)
    max_queue_units: int = Field(4096, ge=1)
    request_timeout_s: float = Field(8.0, gt=0.0)

    # Request limits.
    max_passages: int = Field(256, ge=1)
    max_passage_chars: int = Field(20000, ge=1)
    max_query_chars: int = Field(2000, ge=1)
    max_body_bytes: int = Field(8 * 1024 * 1024, ge=1024)

    warmup: bool = True
