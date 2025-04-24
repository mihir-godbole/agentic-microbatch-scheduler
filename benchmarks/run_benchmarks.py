"""Benchmark harness: replays trace JSON vs scheduler and baseline."""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import time
from typing import List

from inference_client import create_client
from scheduler.scheduler import MicroBatchScheduler
from scheduler.request_collector import Request


async def replay_trace(trace: List[dict], scheduler: MicroBatchScheduler):
    results = []
    start = time.perf_counter()
    latency_samples = []

    async def push(req_dict):
        req = Request(
            prompt=req_dict["prompt"],
            step_id=req_dict.get("step_id", 0),
            trace_id=req_dict.get("trace_id", "bench"),
            priority=req_dict.get("priority", 0),
        )
        start_ts = time.perf_counter()
        await scheduler.submit(req)
        out = await req.future  # wait
        latency_samples.append((time.perf_counter() - start_ts) * 1000)
        results.append(out)

    await asyncio.gather(*(push(r) for r in trace))
    total = time.perf_counter() - start
    return results, latency_samples, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path, help="JSON list of prompts")
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text())
    assert isinstance(trace, list)

    runner = create_client("dummy").run_batch
    scheduler = MicroBatchScheduler(runner, max_batch_size=8, batch_timeout_ms=4)

    loop = asyncio.get_event_loop()
    loop.create_task(scheduler.run())

    res, latencies, total = loop.run_until_complete(replay_trace(trace, scheduler))
    p50 = statistics.quantiles(latencies, n=100)[49]
    p99 = statistics.quantiles(latencies, n=100)[98]
    print(f"P50={p50:.2f}ms P99={p99:.2f}ms total={total:.2f}s")


if __name__ == "__main__":
    main()
