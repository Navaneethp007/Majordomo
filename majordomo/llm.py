"""The brain — an OpenAI-compatible chat endpoint, pointed at OpenRouter.

Lifted from Voicelog's ``llm.py`` — one POST and a typed exception per failure
mode, so the caller can decide between "warn and degrade" and "stop".

Two things have since been added, both learned here.

**``model`` is an argument, not a config field.** Majordomo runs the same
endpoint under four roles — worker, fuser, reducer, chat — and the point of the
config layout is that each role's model swaps without touching code.

**The retry policy distinguishes what is worth retrying.** Voicelog's version
tried twice with no delay, which turns out to be the same as trying once:
OpenRouter's free models sit behind a shared provider pool, and its 429 carries
``retry-after: 5``. Retrying instantly against that is guaranteed to fail, so
every 429 was effectively a single attempt with a cosmetic second. We now wait
what the server asks for, and only for statuses a wait can actually fix — a 403
("only available on agentic harnesses") and a 404 are permanent, and retrying
them just adds latency to a certain failure.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

import httpx

from dataclasses import dataclass

from majordomo.config import BrainConfig


class MissingApiKey(Exception):
    """The configured API key env var is not set. Unrecoverable — we stop."""


class LLMError(Exception):
    """A request failed after all retries. Recoverable — the caller degrades."""


class QuotaExhausted(LLMError):
    """The account's free allowance is spent. Waiting is the only remedy.

    An ``LLMError`` subclass so every existing caller still degrades rather than
    crashing, but distinguished because the response differs: there is nothing
    to retry and no other model to try, so the only useful thing to do is say
    when it comes back.
    """


#: How many times to send the request in total.
MAX_ATTEMPTS = 3

#: Longest we will ever wait between attempts. The briefing runs unattended on
#: wake, so a few seconds is invisible and a minute is a scheduled task that
#: looks hung — past this, degrading to the raw per-source list is the better
#: answer than waiting.
MAX_BACKOFF_SECONDS = 8.0

#: Statuses worth trying again. 429 is the important one, and it arrives in two
#: quite different flavours that need telling apart:
#:
#: - ``upstream_provider_shared_pool`` — the provider is busy right now. Carries
#:   ``retry-after: 5``, and waiting genuinely works.
#: - ``openrouter_free_tier_daily`` — the *account* is out of free requests
#:   until midnight UTC. Nothing inside one call will clear it, and no other
#:   model will either, because the cap is on the account.
#:
#: Only the first is worth a retry. See ``_daily_limit``.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Markers OpenRouter uses for the account-wide daily allowance.
_DAILY_MARKERS = ("free-models-per-day", "openrouter_free_tier_daily")


#: One pooled client for the process, created on first use.
#:
#: ``httpx.post`` opens a fresh connection every call, which means a TCP
#: handshake and a TLS negotiation per request — measured at ~167ms against
#: OpenRouter. That is invisible for a briefing (four requests) and is not for
#: the agent, which makes up to 24 in a row for a single task and paid it every
#: time. Pooling turns that into one handshake per process.
#:
#: Never explicitly closed: this is a short-lived CLI, and the interpreter tears
#: the sockets down on exit. Adding lifecycle management would mean every entry
#: point owning a context manager, for no benefit at this lifespan.
_client: httpx.Client | None = None


def _post(url: str, headers: dict, json: dict, timeout: float) -> httpx.Response:
    """POST through the shared client. The seam tests patch."""
    global _client
    if _client is None:
        _client = httpx.Client(follow_redirects=True)
    return _client.post(url, headers=headers, json=json, timeout=timeout)


def _daily_limit(response: httpx.Response) -> str:
    """A human sentence if this is the daily cap, empty string otherwise.

    Reads the reset stamp out of the body rather than guessing, because "try
    again later" is not actionable and "back at 05:30" is.
    """
    body = response.text[:2000]
    if not any(marker in body for marker in _DAILY_MARKERS):
        return ""

    when = ""
    match = re.search(r'"X-RateLimit-Reset"\s*:\s*"?(\d+)"?', body)
    if match:
        try:
            stamp = datetime.fromtimestamp(int(match.group(1)) / 1000, timezone.utc)
            when = f" It resets at {stamp.astimezone().strftime('%H:%M on %d %b')}."
        except (ValueError, OSError, OverflowError):
            when = ""

    return (
        "Out of free requests for today on this OpenRouter account." + when
        + " Adding credits raises the daily allowance; otherwise wait for the reset."
    )


def _retry_after(response: httpx.Response, attempt: int) -> float:
    """How long to wait before trying again.

    Prefer the server's own ``retry-after`` — it knows when its pool frees up
    and we do not. Fall back to exponential backoff when it says nothing.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), MAX_BACKOFF_SECONDS)
        except ValueError:
            # `retry-after` may legally be an HTTP date. Rather than parse one,
            # fall through to backoff — the date form is rare and the cost of
            # ignoring it is one extra second of waiting.
            pass
    return min(2.0**attempt, MAX_BACKOFF_SECONDS)


@dataclass(frozen=True)
class ToolCall:
    """One tool the model wants run."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    """What came back: text, tool calls, or both.

    Returned instead of a bare string because an agent turn genuinely has two
    possible shapes and the caller must branch on which. ``raw`` is the message
    exactly as the API sent it, because the next request has to echo it back
    verbatim — reconstructing it loses provider-specific fields and the
    conversation stops making sense to the model.
    """

    text: str
    tool_calls: list[ToolCall]
    raw: dict


def complete_with_tools(
    messages: list[dict],
    brain: BrainConfig,
    model: str,
    tools: list[dict],
) -> Reply:
    """A completion that may answer with tool calls instead of prose.

    Shares the retry and fallback layers with ``complete`` — a busy pool is a
    busy pool whether or not tools are in the request.
    """
    payload_extra = {"tools": tools} if tools else {}
    raw = _request(messages, brain, model, payload_extra)

    message = raw.get("message") or {}
    calls: list[ToolCall] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            # A malformed argument blob is the model's mistake, not a transport
            # failure. Pass it through so the agent can hand back an error the
            # model can read and correct, rather than dying here.
            arguments = {"__malformed__": function.get("arguments")}
        if not isinstance(arguments, dict):
            arguments = {"__malformed__": arguments}
        calls.append(
            ToolCall(
                id=str(call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=arguments,
            )
        )

    content = message.get("content")
    return Reply(
        text=content if isinstance(content, str) else "",
        tool_calls=calls,
        raw=message,
    )


def complete(messages: list[dict], brain: BrainConfig, model: str) -> str:
    """Send a chat completion and return the assistant's content.

    Tries ``model``; if that fails in a way retrying could not fix, tries
    ``brain.fallback_model`` once. Every caller gets this without knowing about
    it, which is the point — the choice of *which* model is a config decision
    and the fact that free pools go busy is not something four call sites should
    each have to handle.

    Args:
        messages: OpenAI-format message list.
        brain:    the BrainConfig section — base_url, key env, timeout, fallback.
        model:    the model id for *this role* (worker / fuser / reducer / chat).

    Raises:
        MissingApiKey: the key env var named in config is empty.
        LLMError:      the primary and the fallback both failed.
    """
    choice = _request(messages, brain, model, {})
    content = (choice.get("message") or {}).get("content")
    # Free-tier models return `content: null` on a filtered or empty completion.
    # Returning that verbatim pushed an AttributeError up through callers that
    # only guard against LLMError — a single null answer took down the whole
    # briefing rather than degrading one source. Normalising here covers every
    # caller at once.
    return content if isinstance(content, str) else ""


def _request(
    messages: list[dict], brain: BrainConfig, model: str, extra: dict
) -> dict:
    """The shared request layer: retries, then the fallback model.

    Returns the raw ``choices[0]`` so a caller that needs tool calls can read
    them and one that only wants prose can ignore them. Both go through the same
    retry and fallback path, because a busy pool is a busy pool either way.
    """
    try:
        return _request_one(messages, brain, model, extra)
    except QuotaExhausted:
        # The cap is on the account, so the fallback is exactly as exhausted as
        # the primary. Trying it spends nothing but time and turns one clear
        # sentence into two copies of it.
        raise
    except LLMError as primary:
        fallback = (brain.fallback_model or "").strip()
        if not fallback or fallback == model:
            raise
        try:
            return _request_one(messages, brain, fallback, extra)
        except LLMError as secondary:
            raise LLMError(
                f"{model} failed ({primary}); fallback {fallback} also failed "
                f"({secondary})"
            ) from secondary


def _request_one(
    messages: list[dict], brain: BrainConfig, model: str, extra: dict
) -> dict:
    """One model, with retries. See ``_request`` for the fallback layer."""
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
        **extra,
    }

    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = _post(url, headers=headers, json=payload, timeout=brain.timeout)
            if not response.is_success:
                last_exc = Exception(f"HTTP {response.status_code}: {response.text[:300]}")

                # Permanent failures are not worth a second look. A 403 on
                # OpenRouter means the model is gated to partner apps
                # ("only available on agentic harnesses"), a 404 means the id is
                # wrong, a 401 means the key is — none of which a retry fixes.
                # Retrying them only adds latency to a certain failure.
                if response.status_code not in RETRYABLE_STATUS:
                    break

                # A daily quota is a 429 that no amount of waiting inside this
                # call will clear, and no other model will dodge — the cap is on
                # the account, not the model. Retrying it burns six requests
                # against a limit you have already hit, then reports it as a
                # wall of JSON. Raised immediately, and past the fallback layer,
                # because a fallback is exactly as capped as the primary.
                exhausted = _daily_limit(response)
                if exhausted:
                    raise QuotaExhausted(exhausted)

                if attempt < MAX_ATTEMPTS - 1:
                    # This is the whole point. The old policy retried
                    # immediately, so against a `retry-after: 5` the second
                    # attempt was guaranteed to fail too — every 429 was
                    # effectively one attempt with a cosmetic second.
                    time.sleep(_retry_after(response, attempt))
                continue

            choice = response.json()["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("choices[0] was not an object")
            return choice
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            # httpx.HTTPError: network/transport failures.
            # ValueError/KeyError/IndexError/TypeError: a 2xx whose body isn't
            # the expected OpenAI shape — a proxy error envelope returned as 200,
            # say. Both are worth another go, then we give up so the caller can
            # fall back to the raw per-source list.
            last_exc = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(min(2.0**attempt, MAX_BACKOFF_SECONDS))

    raise LLMError(str(last_exc))
