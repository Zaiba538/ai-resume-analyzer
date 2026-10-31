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
        reply = self.replies.popleft()
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
