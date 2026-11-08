"""Model wrapper: mini-swe-agent's LiteLLM model, calling the API the way Scale's SWE-agent does.

Same as Scale (SWE-agent sweagent/agent/models.py, LiteLLMModel._single_query): Scale's tool
definitions, temperature and top_p, api_base, `user=LITELLM_USER`, and the input-token limit counted
locally with litellm's token_counter. Messages sent to the API carry only role, content, tool_calls
and tool_call_id, as Scale's _history_to_messages does.
"""

import logging
import os
import time

import litellm
from minisweagent.models.litellm_model import LitellmModel

from hilx.scale import ROOT
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("hilx.model")

API_KEYS = ("role", "content", "tool_calls", "tool_call_id")
PRICES = ROOT / "configs" / "model_prices.json"  # prices for models LiteLLM doesn't know


class ContextLimitExceeded(Exception):
    pass


class HilModel(LitellmModel):
    def __init__(self, scale_model_config, tools: list[dict], routing: dict | None = None):
        """scale_model_config: the `model` part of a Scale agent config (DefaultSQLAgentConfig.model).
        routing: configs/routing/*.yaml, ours, not Scale's: an OpenRouter provider pin and a call timeout."""
        cfg = scale_model_config
        kwargs = {"temperature": cfg.temperature, "user": os.environ["LITELLM_USER"].strip()}
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        routing = routing or {}
        if routing.get("provider"):
            kwargs["extra_body"] = {"provider": routing["provider"]}
        if routing.get("timeout_s"):
            kwargs["timeout"] = routing["timeout_s"]
        super().__init__(model_name=cfg.name, model_kwargs=kwargs, litellm_model_registry=str(PRICES))
        self.tools = tools
        self.max_input_tokens = cfg.max_input_tokens

    def _query(self, messages: list[dict], **kwargs):
        return litellm.completion(model=self.config.model_name, messages=messages, tools=self.tools, **(self.config.model_kwargs | kwargs))

    def query(self, messages: list[dict], **kwargs) -> dict:
        """Returns an assistant message in Scale's shape: content, and tool_calls as plain dicts (not parsed here)."""
        api_messages = [{k: v for k, v in m.items() if k in API_KEYS} for m in messages]
        n_tokens = litellm.utils.token_counter(messages=api_messages, model=self.config.model_name)
        if self.max_input_tokens and n_tokens > self.max_input_tokens > 0:
            raise ContextLimitExceeded(f"Input tokens {n_tokens} exceed max tokens {self.max_input_tokens}")
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(api_messages, **kwargs)
        cost = self._calculate_cost(response)["cost"]
        msg = response.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.to_dict() for tc in msg.tool_calls] if msg.tool_calls else [],
            "extra": {"cost": cost, "input_tokens": n_tokens, "timestamp": time.time(), "response": response.model_dump(mode="json")},
        }
