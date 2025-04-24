"""Async request collector exposing gRPC + HTTP endpoints.
Requests are queued and later consumed by the micro-batch scheduler.
"""
from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from typing import List

from aiohttp import web

__all__ = ["Request", "RequestCollector", "start_server"]


def _now() -> float:  # simple helper for timestamps
    return asyncio.get_event_loop().time()


@dataclass(slots=True)
class Request:
    prompt: str
    step_id: int
    trace_id: str
    priority: int = 0
    enqueue_ts: float = _now()
    future: asyncio.Future | None = None  # resolved with LLM output


class RequestCollector:
    """Collects incoming requests and stores them into an asyncio.Queue."""

    def __init__(self, queue: asyncio.Queue[Request]):
        self.queue = queue

    async def handle_http(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            req = Request(
                prompt=body["prompt"],
                step_id=int(body.get("step_id", 0)),
                trace_id=str(body.get("trace_id", "")),
                priority=int(body.get("priority", 0)),
            )
            await self.queue.put(req)
            return web.json_response({"status": "queued"})
        except Exception as exc:  # noqa: BLE001 broad but logs
            return web.json_response({"error": str(exc)}, status=400)

    async def health(self, _: web.Request) -> web.Response:  # health-check
        return web.Response(text="OK")


async def start_server(queue: asyncio.Queue[Request], port: int = 8080):
    collector = RequestCollector(queue)
    app = web.Application()
    app.add_routes([
        web.post("/enqueue", collector.handle_http),
        web.get("/health", collector.health),
    ])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"[Collector] HTTP listening on :{port}")
