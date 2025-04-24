import asyncio

import pytest

from scheduler.request_collector import Request
from scheduler.scheduler import MicroBatchScheduler


@pytest.mark.asyncio
async def test_batching():
    outs = []

    async def echo(prompts):
        await asyncio.sleep(0.01)
        return prompts

    sched = MicroBatchScheduler(echo, max_batch_size=4, batch_timeout_ms=10)
    loop_task = asyncio.create_task(sched.run())

    reqs = [Request(prompt=str(i), step_id=i, trace_id="t") for i in range(5)]
    for r in reqs:
        await sched.submit(r)

    outs = await asyncio.gather(*(r.future for r in reqs))

    assert outs == [str(i) for i in range(5)]

    await sched.shutdown()
    await loop_task
