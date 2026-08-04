"""The brain — an OpenAI-compatible chat endpoint, pointed at OpenRouter.

Lifted almost verbatim from Voicelog's ``llm.py``, which already does exactly
what is needed: one POST, retry once, and a typed exception per failure mode so
the caller can decide between "warn and degrade" and "stop".

The one addition is ``model`` as an explicit argument rather than a config
field. Majordomo runs the same endpoint under three roles — fuser, reducer,
worker — and the whole point of the config layout is that each role's model can
be swapped without touching code.
"""
from __future__ import annotations

import os

import httpx

from majordomo.config import BrainConfig


class MissingApiKey(Exception):
    """The configured API key env var is not set. Unrecoverable — we stop."""


class LLMError(Exception):
    """A request failed after all retries. Recoverable — the caller degrades."""


def complete(messages: list[dict], brain: BrainConfig, model: str) -> str:
    """Send a chat completion and return the assistant's content.

    Args:
        messages: OpenAI-format message list.
        brain:    the BrainConfig section — base_url, key env, timeout.
        model:    the model id for *this role* (fuser / reducer / worker).

    Raises:
        MissingApiKey: the key env var named in config is empty.
        LLMError:      both attempts failed, or the body wasn't the expected shape.
    """
    api_key = os.environ.get(brain.api_key_env)
    if not api_key:
        raise MissingApiKey(
            f"Set the {brain.api_key_env} environment variable with your "
            f"{brain.provider} API key."
        )

    url = f"{brain.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        # OpenRouter uses these for attribution on free-tier models.
        "HTTP-Referer": "https://github.com/Navaneethp007/majordomo",
        "X-Title": "Majordomo",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": brain.temperature,
        "max_tokens": 4096,
    }

    last_exc: Exception | None = None
    for _attempt in range(2):  # try once, retry once
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=brain.timeout)
            if not response.is_success:
                last_exc = Exception(f"HTTP {response.status_code}: {response.text[:300]}")
                continue
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            # httpx.HTTPError: network/transport failures.
            # ValueError/KeyError/IndexError/TypeError: a 2xx whose body isn't
            # the expected OpenAI shape — a proxy error envelope returned as 200,
            # say. Treat like any other failure: retry once, then give up so the
            # caller can fall back to the raw per-source list.
            last_exc = exc

    raise LLMError(str(last_exc))
