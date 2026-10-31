"""Structured LLM output: strict JSON schema, local validation, one repair retry.

Flow for one call:

1. A "wire" JSON schema is derived from the Pydantic model: references inlined,
   every object closed (``additionalProperties: false``) with all properties
   required, and value constraints (lengths, counts, patterns) removed. Groq's
   strict mode supports only a subset of JSON Schema, so the wire schema
   carries structure and enums while the constraints are enforced locally.
2. The model is called with ``response_format=json_schema`` (strict). The SDK
   handles transport retries and timeouts (``backend.llm.client``).
3. The JSON is validated against the full Pydantic model. If it is malformed,
   violates the schema or limits, or Groq reports ``json_validate_failed``, the
   call is repeated once with a short repair instruction (``MAX_REPAIRS``).
   A second failure raises ``InvalidLLMOutputError`` (502).
4. The validated output goes through ``validate_llm_output``: strings are
   sanitised and links not present in the source text are neutralised.

Every outcome is reported to the upstream monitor as a passive signal. Logs
carry model, role, token counts, attempts and duration, never prompts, outputs,
failed generations or keys.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import groq
from pydantic import BaseModel, ValidationError

from backend.llm.client import LLMClients, LLMRole
from backend.llm.errors import (
    InvalidLLMOutputError,
    LLMError,
    is_schema_generation_failure,
    map_groq_error,
    monitor_code,
)
from backend.llm.prompts import Message
from backend.schemas.security import SecurityFlag
from backend.security.output_validation import validate_llm_output

logger = logging.getLogger("backend.llm")

MAX_REPAIRS = 1
REPAIR_INSTRUCTION = (
    "Your previous response was not valid for the required JSON schema or its limits. "
    "Respond again with only a JSON object that satisfies the schema exactly."
)
_CONSTRAINT_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "default",
        "title",
    }
)


def _inline_refs(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_refs(copy.deepcopy(definitions[name]), definitions)
        return {k: _inline_refs(v, definitions) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(item, definitions) for item in node]
    return node


def _to_wire(node: Any) -> Any:
    if isinstance(node, list):
        return [_to_wire(item) for item in node]
    if not isinstance(node, dict):
        return node
    wire: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties":
            # Keys here are field names, not keywords: never filter them.
            wire[key] = {name: _to_wire(sub) for name, sub in value.items()}
        elif key not in _CONSTRAINT_KEYWORDS:
            wire[key] = _to_wire(value)
    if wire.get("type") == "object" or "properties" in wire:
        wire["type"] = "object"
        wire["additionalProperties"] = False
        wire["required"] = list(wire.get("properties", {}))
    return wire


@lru_cache(maxsize=32)
def _wire_schema_cached(schema: type[BaseModel]) -> dict[str, Any]:
    full = schema.model_json_schema()
    inlined = _inline_refs(full, full.get("$defs", {}))
    wire = _to_wire(inlined)
    wire["title"] = schema.__name__
    return wire


def wire_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Strict-mode-compatible JSON schema sent to the provider (a fresh copy)."""
    return copy.deepcopy(_wire_schema_cached(schema))


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    output: T
    flags: tuple[SecurityFlag, ...] = ()
    attempts: int = 1
    usage: Usage = field(default_factory=Usage)


def _usage(raw: Any) -> Usage:
    metadata = getattr(raw, "usage_metadata", None) or {}
    return Usage(
        input_tokens=int(metadata.get("input_tokens", 0) or 0),
        output_tokens=int(metadata.get("output_tokens", 0) or 0),
    )


class StructuredLLM:
    """Runs schema-constrained generations for agent roles."""

    def __init__(self, clients: LLMClients) -> None:
        self._clients = clients

    async def generate[T: BaseModel](
        self,
        role: LLMRole,
        schema: type[T],
        messages: Sequence[Message],
        *,
        source_texts: Iterable[str],
    ) -> StructuredResult[T]:
        model = self._clients.chat_model(role)
        runnable = model.with_structured_output(
            wire_schema(schema), method="json_schema", strict=True, include_raw=True
        )
        conversation = list(messages)
        sources = list(source_texts)
        monitor = self._clients.monitor
        started = time.perf_counter()
        input_tokens = output_tokens = 0

        for attempt in range(1, MAX_REPAIRS + 2):
            try:
                result = await runnable.ainvoke(conversation)
            except groq.BadRequestError as exc:
                if not is_schema_generation_failure(exc):
                    raise self._fail(role, map_groq_error(exc), started, attempt) from None
                if monitor is not None:
                    monitor.record_request_outcome(True)
                conversation.append(("user", REPAIR_INSTRUCTION))
                continue
            except groq.GroqError as exc:
                raise self._fail(role, map_groq_error(exc), started, attempt) from None

            if monitor is not None:
                monitor.record_request_outcome(True)
            usage = _usage(result.get("raw"))
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens

            parsed = result.get("parsed")
            if result.get("parsing_error") is None and isinstance(parsed, dict):
                try:
                    validated = schema.model_validate(parsed)
                    output, flags = validate_llm_output(validated, source_texts=sources)
                except ValidationError:
                    pass
                else:
                    self._log(role, started, attempt, input_tokens, output_tokens, None)
                    return StructuredResult(
                        output=output,
                        flags=tuple(flags),
                        attempts=attempt,
                        usage=Usage(input_tokens, output_tokens),
                    )
            conversation.append(("user", REPAIR_INSTRUCTION))

        raise self._fail(role, InvalidLLMOutputError(), started, MAX_REPAIRS + 1, report=False)

    def _fail(
        self, role: LLMRole, error: LLMError, started: float, attempts: int, *, report: bool = True
    ) -> LLMError:
        if report and self._clients.monitor is not None:
            self._clients.monitor.record_request_outcome(False, monitor_code(error))
        self._log(role, started, attempts, 0, 0, error.code)
        return error

    def _log(
        self,
        role: LLMRole,
        started: float,
        attempts: int,
        input_tokens: int,
        output_tokens: int,
        error_code: str | None,
    ) -> None:
        logger.info(
            "llm_call_completed",
            extra={
                "node": role,
                "model": self._clients.model_for(role),
                "timing_ms": round((time.perf_counter() - started) * 1000, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "attempts": attempts,
                "error_code": error_code,
            },
        )
