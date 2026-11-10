"""Shared Pydantic base class for every model that crosses a boundary."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Immutable model that rejects unknown fields, so typos fail loudly at the boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)
