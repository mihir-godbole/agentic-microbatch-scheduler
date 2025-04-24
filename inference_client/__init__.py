"""Inference client abstraction."""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

__all__ = ["BaseClient", "create_client"]


@runtime_checkable
class BaseClient(Protocol):
    async def run_batch(self, prompts: List[str]) -> List[str]:
        ...


def create_client(kind: str = "dummy") -> BaseClient:
    if kind == "dummy":
        from .triton_client import DummyClient

        return DummyClient()
    elif kind == "triton":
        from .triton_client import TritonClient

        return TritonClient()
    raise ValueError(kind)
