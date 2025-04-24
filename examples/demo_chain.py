"""Demo script to show micro-batching in action."""
import asyncio
import random

from inference_client import create_client
from scheduler.scheduler import MicroBatchScheduler
from scheduler.request_collector import Request


async def main():
    runner = create_client("dummy").run_batch
    scheduler = MicroBatchScheduler(runner, max_batch_size=8, batch_timeout_ms=5)

    # Start scheduler loop
    loop_task = asyncio.create_task(scheduler.run())

    # Feed random requests
    for i in range(100):
        req = Request(
            prompt=f"hello {i}", step_id=i, trace_id="demo", priority=random.randint(0, 5)
        )
        await scheduler.submit(req)
        await asyncio.sleep(random.uniform(0.0, 0.002))

    await asyncio.sleep(1)
    await scheduler.shutdown()
    await loop_task


if __name__ == "__main__":
    asyncio.run(main())
