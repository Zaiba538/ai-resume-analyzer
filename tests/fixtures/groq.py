"""In-process fake of Groq's OpenAI-compatible chat-completions endpoint.

Used through ``httpx.MockTransport`` so that the real ChatGroq and groq SDK
code runs: request building, retries with backoff, timeouts and error mapping.
Responses are scripted per call; every request is recorded for assertions.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Reply:
    """One scripted reply: chat content, an HTTP error, or a transport exception."""

    content: str | None = None
    status: int = 200
    error_code: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    exception: Exception | None = None
    usage: tuple[int, int] = (120, 40)


def json_reply(payload: Any) -> Reply:
    return Reply(content=json.dumps(payload, ensure_ascii=False))


class FakeGroqChat:
    def __init__(self) -> None:
        self.replies: deque[Reply] = deque()
        self.requests: list[httpx.Request] = []

    def queue(self, *replies: Reply) -> FakeGroqChat:
        self.replies.extend(replies)
        return self

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("FakeGroqChat: no scripted reply left")
        return self.respond(request, self.replies.popleft())

    def respond(self, request: httpx.Request, reply: Reply) -> httpx.Response:
        if reply.exception is not None:
            raise reply.exception
        headers = {"retry-after-ms": "1", **reply.headers}
        if reply.status != 200:
            error = {"message": "scripted failure", "type": "invalid_request_error"}
            if reply.error_code:
                error["code"] = reply.error_code
                error["failed_generation"] = "SECRET-FAILED-GENERATION"
            return httpx.Response(reply.status, json={"error": error}, headers=headers)
        body = json.loads(request.content)
        prompt_tokens, completion_tokens = reply.usage
        return httpx.Response(
            200,
            headers=headers,
            json={
                "id": f"chatcmpl-fake-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


class RoutedFakeGroq(FakeGroqChat):
    """Replies chosen by the request's agent prompt, not by arrival order.

    For workflows whose agents run in parallel. Each route has its own reply
    queue; Prompt Guard requests are recognised by model and answered with
    ``guard_score``. ``routes`` records the route of every request in order.
    """

    PREFIXES = {
        "resume": "You are a meticulous resume analyst.",
        "interview": "You are an experienced technical interviewer.",
        "skill_gap": "You are a careful career advisor.",
        "critic": "You are a rigorous reviewer of AI-generated resume analysis.",
    }

    def __init__(self, guard_score: str = "0.01") -> None:
        super().__init__()
        self.guard_score = guard_score
        self.queues: dict[str, deque[Reply]] = {name: deque() for name in self.PREFIXES}
        self.routes: list[str] = []

    def script(self, route: str, *replies: Reply) -> RoutedFakeGroq:
        self.queues[route].extend(replies)
        return self

    @staticmethod
    def route_of(body: dict[str, Any]) -> str:
        if "prompt-guard" in body.get("model", ""):
            return "prompt_guard"
        system = body["messages"][0]["content"]
        for name, prefix in RoutedFakeGroq.PREFIXES.items():
            if system.startswith(prefix):
                return name
        raise AssertionError("RoutedFakeGroq: unrecognised request")

    def bodies_for(self, route: str) -> list[dict[str, Any]]:
        return [body for body, name in zip(self.bodies, self.routes, strict=True) if name == route]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.route_of(json.loads(request.content))
        self.routes.append(route)
        if route == "prompt_guard":
            return self.respond(request, Reply(content=self.guard_score))
        if not self.queues[route]:
            raise AssertionError(f"RoutedFakeGroq: no scripted reply left for {route}")
        return self.respond(request, self.queues[route].popleft())
