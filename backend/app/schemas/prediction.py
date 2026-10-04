"""Public request bodies. Clinical values stay raw; model features are not accepted."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_OBSERVATIONS = 336
MAX_VALUES = 40


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, float | None]
    hour: int | None = None

    @field_validator("values")
    @classmethod
    def bound_values(cls, values: dict[str, float | None]) -> dict[str, float | None]:
        if len(values) > MAX_VALUES:
            raise ValueError("an observation contains more parameters than the supported list")
        return values


class SequenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[Observation] = Field(min_length=1, max_length=MAX_OBSERVATIONS)
