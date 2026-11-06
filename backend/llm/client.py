"""Groq clients: one configured chat model per agent role, plus a raw client.

* One shared ``httpx.AsyncClient`` backs every Groq call and is closed on
  application shutdown.
* Each role uses the model configured for it (``GROQ_MODEL_<ROLE>``) with the
  configured timeout, SDK retry count (exponential backoff on connection
  errors, 408/409/429 and 5xx, honouring ``Retry-After``), output-token cap and
  temperature 0. ``reasoning_effort`` is sent only to gpt-oss reasoning models.
* Models are given no tools. Text from users only ever reaches them as data
  (see ``backend.llm.prompts``).
* Without an API key no client is created; using one raises
  ``LLMNotConfiguredError`` (503).
"""

from __future__ import annotations

from typing import Literal

import groq
import httpx
from langchain_groq import ChatGroq

from backend.core.config import Settings
from backend.health.upstream import GroqUpstreamMonitor
from backend.llm.errors import LLMNotConfiguredError

LLMRole = Literal["resume", "skill_gap", "critic", "interview", "evaluator"]

_ROLE_SETTINGS: dict[LLMRole, str] = {
    "resume": "groq_model_resume",
    "skill_gap": "groq_model_skill_gap",
    "critic": "groq_model_critic",
    "interview": "groq_model_interview",
    "evaluator": "groq_model_evaluator",
}
REASONING_MODEL_PREFIX = "openai/gpt-oss"


class LLMClients:
    """Application-scoped Groq clients with an explicit lifecycle."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        monitor: GroqUpstreamMonitor | None = None,
    ) -> None:
        self.settings = settings
        self.monitor = monitor
        self._owns_http_client = http_client is None
        self._http_client = http_client
        self._chat_models: dict[LLMRole, ChatGroq] = {}
        self._raw_client: groq.AsyncGroq | None = None

    @property
    def configured(self) -> bool:
        return self.settings.groq_api_key is not None

    def model_for(self, role: LLMRole) -> str:
        return str(getattr(self.settings, _ROLE_SETTINGS[role]))

    def _http(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    def _require_key(self) -> str:
        if self.settings.groq_api_key is None:
            raise LLMNotConfiguredError()
        return self.settings.groq_api_key.get_secret_value()

    def chat_model(self, role: LLMRole) -> ChatGroq:
        """The configured chat model for an agent role (created once)."""
        if role not in self._chat_models:
            api_key = self._require_key()
            model = self.model_for(role)
            options: dict[str, object] = {}
            if model.startswith(REASONING_MODEL_PREFIX):
                options["reasoning_effort"] = self.settings.llm_reasoning_effort
            self._chat_models[role] = ChatGroq(
                model=model,
                api_key=api_key,
                temperature=0,
                max_tokens=self.settings.llm_max_output_tokens,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
                http_async_client=self._http(),
                **options,
            )
        return self._chat_models[role]

    def raw_client(self) -> groq.AsyncGroq:
        """Plain Groq client (used for the Prompt Guard classifier)."""
        if self._raw_client is None:
            self._raw_client = groq.AsyncGroq(
                api_key=self._require_key(),
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
                http_client=self._http(),
            )
        return self._raw_client

    async def aclose(self) -> None:
        self._chat_models.clear()
        self._raw_client = None
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
        self._http_client = None
