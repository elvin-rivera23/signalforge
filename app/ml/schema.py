from typing import Literal

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    symbol: str = Field(default="AAPL")
    interval: str = Field(default="5m")
    lookahead_n: int = Field(default=3, ge=1)
    up_threshold: float = Field(default=0.001, ge=0.0)
    start: str | None = None
    end: str | None = None
    dataset_version: str = Field(default="v0")
    synthetic: bool = Field(default=True)
    synthetic_mode: Literal["up", "down", "sideways"] | None = "up"
