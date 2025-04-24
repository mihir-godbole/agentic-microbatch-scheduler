"""Dynamic micro-batch scheduler.
Formulates GPU-friendly batches from asynchronous requests.
Includes optional CUDA kernel to profile overhead of launch.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

from .request_collector import Request
from .cuda_utils import batch_stats

log = logging.getLogger(__name__)


class MicroBatchScheduler:
    def __init__(
        self,
        downstream_runner,  # callable: List[str] -> List[str]
        *,
        max_batch_size: int = 16,
        max_batch_tokens: int | None = None,
        batch_timeout_ms: int = 4,
    ) -> None:
        self.queue: asyncio.Queue[Request] = asyncio.Queue()
        self.downstream_runner = downstream_runner
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.batch_timeout = batch_timeout_ms / 1000.0
        self._shutdown = asyncio.Event()

    async def run(self):
        """Main loop. Builds and dispatches batches."""
        while not self._shutdown.is_set():
            try:
                batch: List[Request] = []
                # Wait for first item w/ timeout
                try:
                    first = await asyncio.wait_for(
                        self.queue.get(), timeout=self.batch_timeout
                    )
                    if first.future is None:
                        first.future = asyncio.get_running_loop().create_future()
                    batch.append(first)
                except asyncio.TimeoutError:
                    continue  # nothing available

                # non-blocking drain until constraints met
                while len(batch) < self.max_batch_size:
                    try:
                        req = self.queue.get_nowait()
                        if req.future is None:
                            req.future = asyncio.get_running_loop().create_future()
                        batch.append(req)
                    except asyncio.QueueEmpty:
                        break
                # Launch downstream asynchronously
                prompts = [r.prompt for r in batch]
                asyncio.create_task(self._execute_batch(prompts, batch))
            except Exception as exc:  # pragma: no cover
                log.exception("Scheduler loop error: %s", exc)

    async def _execute_batch(self, prompts: List[str], reqs: List[Request]):
        import time, torch

        tic = time.perf_counter()
        outputs = await self.downstream_runner(prompts)
        toc = time.perf_counter()

        # Compute prompt length statistics on GPU using the custom CUDA kernel
        lengths = torch.tensor([len(p) for p in prompts], dtype=torch.int32, device="cuda")
        total_tokens, max_tokens = batch_stats(lengths)

        log.info(
            "batch_size=%d tok_sum=%d tok_max=%d latency=%.3fms",
            len(prompts),
            total_tokens,
            max_tokens,
            (toc - tic) * 1000,
        )
        # Resolve futures preserving original order
        for req, out in zip(reqs, outputs):
            if req.future and not req.future.done():
                req.future.set_result(out)

    async def shutdown(self):
        self._shutdown.set()

    # ---------------- Convenience API -----------------
    async def submit(self, request: Request):
        await self.queue.put(request)


# ---------------- Example CUDA helper -----------------
import torch

def dummy_cuda_kernel(n: int = 1024):
    """A very small CUDA kernel to measure launch overhead."""

    code = r"""
    extern "C" __global__ void add_kernel(float* x, float val, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) x[idx] += val;
    }
    """

    from torch.utils.cpp_extension import load_inline

    module = load_inline(
        name="add_kernel_module",
        cpp_sources="",
        cuda_sources=code,
        functions=["add_kernel"],
        verbose=False,
    )
    x = torch.zeros(n, device="cuda")
    module.add_kernel(x, 1.0, n, grid=(n // 256 + 1, 1, 1), block=(256, 1, 1))
    torch.cuda.synchronize()
    return x[0].item()
