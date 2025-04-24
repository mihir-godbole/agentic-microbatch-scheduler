"""Triton inference client or dummy fallback."""
from __future__ import annotations

import asyncio
from typing import List, Any


class DummyClient:
    async def run_batch(self, prompts: List[str]) -> List[str]:
        await asyncio.sleep(0.01)  # simulate latency
        return [p[::-1] for p in prompts]  # dummy transform


# NOTE: Real Triton implementation simplified for brevity – assumes a model named "llm" that
# takes a string tensor "PROMPT" and returns string tensor "OUTPUT".
# This keeps compatibility with both CPU-only envs and CUDA machines by optional import.


class TritonClient:
    """Thin async wrapper over tritonclient.grpc.InferenceServerClient."""

    def __init__(self, url: str = "localhost:8001", model_name: str = "llm") -> None:
        try:
            import tritonclient.grpc.aio as grpcclient
            from tritonclient.grpc import InferInput, InferRequestedOutput

            self._grpc = grpcclient.InferenceServerClient(url=url)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Triton client init failed; ensure tritonclient[grpc] installed") from exc

        self.model_name = model_name
        self._InferInput = InferInput  # store references for speed
        self._InferRequestedOutput = InferRequestedOutput

    async def run_batch(self, prompts: List[str]) -> List[str]:
        batch_size = len(prompts)

        # Prepare input tensor – batching ragged strings is supported by Triton string type
        inp = self._InferInput("PROMPT", [batch_size], "BYTES")
        inp.set_data_from_numpy(
            # numpy array of byte strings
            import numpy as np; np.asarray(prompts, dtype=object)
        )

        out_spec = self._InferRequestedOutput("OUTPUT")

        resp = await self._grpc.infer(self.model_name, inputs=[inp], outputs=[out_spec])
        out_array: Any = resp.as_numpy("OUTPUT")
        return [x.decode("utf-8") for x in out_array]


# alias
TritonClient = TritonClient
