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
from datetime import datetime, timedelta, timezone

import httpx

from dataclasses import dataclass

from majordomo.config import BrainConfig


class MissingApiKey(Exception):
    """The configured API key env var is not set. Unrecoverable — we stop."""


class LLMError(Exception):
    """A request failed after all retries. Recoverable — the caller degrades.

    ``status`` carries the HTTP status when there was one, so ``_explain`` can
    tell a permanent failure from a transient one without parsing it back out of
    the message. Kept as an attribute rather than threaded through the dozen
    places that display an ``LLMError``: the advice ends up *in* the message, so
    every existing caller shows it without knowing it exists.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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


#: Names for "the whole account is out", as distinct from "this model's pool is
#: busy". There is no structural signal for that difference — a reset stamp
#: cannot say whose limit it is — so this is the one place a provider's own
#: wording genuinely belongs.
#:
#: When it stops matching, the cost is bounded: a far-off limit still stops
#: retrying *this* model and simply tries the fallback once more than it needed
#: to. It is not the six wasted attempts that depending on it for everything
#: used to produce.
_ACCOUNT_WIDE_MARKERS = (
    "free-models-per-day",
    "openrouter_free_tier_daily",
    "per-day",
    "daily limit",
    "daily quota",
)

#: Past this, waiting inside one call is pointless. A shared pool answers
#: ``retry-after: 5``; anything measured in minutes is a different kind of
#: limit. Two minutes sits far above the first and far below a daily reset.
QUOTA_HORIZON_SECONDS = 120

#: The reset stamp, wherever it is written. OpenRouter sends no ``X-RateLimit-*``
#: as real HTTP headers — confirmed against a live response — so the only copy
#: is nested inside ``error.metadata.headers`` in the body, and it has to be
#: read out of the JSON text. Case-insensitive because the casing of a key
#: inside a JSON body is the provider's whim, not a protocol.
_RESET_STAMP = re.compile(r'"x-ratelimit-reset"\s*:\s*"?(\d+)"?', re.IGNORECASE)


def _reset_seconds(response: httpx.Response) -> float | None:
    """How long until this limit clears, or None if nothing says.

    Prefers the real ``retry-after`` header, which is a protocol answer, over
    the reset stamp a provider buried in its JSON. A stamp already in the past
    means the limit has lifted, which is not a reason to stop.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass

    match = _RESET_STAMP.search(response.text[:4000])
    if match is None:
        return None
    try:
        # Milliseconds since the epoch, as OpenRouter sends it.
        stamp = datetime.fromtimestamp(int(match.group(1)) / 1000, timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None
    return max(0.0, (stamp - datetime.now(timezone.utc)).total_seconds())


def _worth_waiting(response: httpx.Response) -> bool:
    """Could this limit clear soon enough to retry inside this call?

    The structural half of the question, and the half a reset stamp can answer.
    Unknown counts as yes: an absent stamp is not evidence of a quota, and
    refusing to retry on no information is the worse mistake.
    """
    seconds = _reset_seconds(response)
    return seconds is None or seconds <= QUOTA_HORIZON_SECONDS


def _daily_limit(response: httpx.Response) -> str:
    """A sentence if the whole *account* is out of requests, else "".

    Two questions, and three versions of this function got the relationship
    between them wrong in three different ways:

    - **Will another model help?** Answered by whose limit it is. Only the
      provider's own wording says that — no reset stamp can. This is the gate.
    - **Will waiting help?** Answered by the reset distance, which is structural
      and survives a rename. It refines the *message*, and rules out the case
      where a "daily" wording arrives on something that clears in seconds.

    The version before this made the distance the gate, and a cap carrying the
    marker but no reset information came back as "" — because unknown distance
    counts as "keep waiting", which is right for retrying and wrong here. A body
    saying ``free-models-per-day`` is a daily cap whether or not anyone said
    when it lifts. That is the stronger evidence, so it goes first.

    The version before *that* made the marker do both jobs, so a long
    ``retry-after`` on one model's pool skipped the fallback that would have
    worked.
    """
    body = response.text[:4000].lower()
    if not any(marker in body for marker in _ACCOUNT_WIDE_MARKERS):
        return ""

    seconds = _reset_seconds(response)

    # A "daily" wording on something clearing in seconds is a provider being
    # loose with words, not a spent quota. Only a *known* short distance
    # overrides the marker; not knowing leaves the marker standing.
    if seconds is not None and seconds <= QUOTA_HORIZON_SECONDS:
        return ""

    when = ""
    if seconds is not None:
        stamp = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        when = f" It clears at {stamp.astimezone().strftime('%H:%M on %d %b')}."

    return (
        "Out of free requests for today on this account." + when
        + " Adding credits raises the daily allowance; otherwise wait for the reset."
    )


#: String codes OpenAI-compatible providers send in place of a status, mapped
#: to the status they mean. Dropping a string code left ``code`` as ``None``,
#: which skipped the non-retryable break — so a certain failure like a bad key
#: spent all six attempts proving it.
_STRING_CODES = {
    "invalid_api_key": 401,
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "permission_denied": 403,
    "model_not_found": 404,
    "not_found_error": 404,
    "context_length_exceeded": 400,
    "invalid_prompt": 400,
    "rate_limit_error": 429,
    "rate_limit_exceeded": 429,
    "overloaded_error": 529,
    "server_error": 500,
    "api_error": 500,
}


def _as_status(code) -> int | None:
    """A provider's error code as an HTTP status, where that is knowable.

    An unrecognised string returns ``None``, which means *retry* — being wrong
    in that direction costs a few seconds, while being wrong the other way turns
    a transient failure into a permanent one.
    """
    if isinstance(code, int):
        return code
    if isinstance(code, str):
        stripped = code.strip()
        if stripped.isdigit():
            return int(stripped)
        return _STRING_CODES.get(stripped.lower())
    return None


def _error_envelope(body) -> tuple[str, int | None] | None:
    """``(message, code)`` if this body is an error wearing a success status.

    OpenRouter answers ``200`` with ``{"error": {...}}`` when an upstream
    provider dies mid-request. Read as a completion that becomes
    ``KeyError('choices')``, whose entire string form is ``"'choices'"`` — the
    least informative possible account of a failure whose cause was sitting
    right there in the body.

    Two things keep it from firing on a good answer, both learned by it doing
    exactly that:

    - **A usable completion wins.** Some providers send ``error: {}`` alongside
      real ``choices``. Treating the key's presence as the signal discarded the
      answer, retried three times, burned the fallback and raised.
    - **Empty is not an error.** ``{}``, ``""`` and ``0`` are all falsy and all
      previously became an "error" whose message was their repr.
    """
    if not isinstance(body, dict):
        return None

    error = body.get("error")
    if not error:
        return None

    # A completion that is actually there beats an error key that says nothing
    # about it. Checked before reading the error, not after.
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return None

    if isinstance(error, dict):
        message = str(error.get("message") or error.get("type") or error)
        return message, _as_status(error.get("code") or error.get("type"))
    return str(error), None


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


def _roles_using(brain: BrainConfig, model: str) -> list[tuple[str, str, str]]:
    """Which of the six roles point at this model id. Read from the config.

    The role is not passed in anywhere, and it does not need to be: ``brain``
    already holds the mapping, so it can be read backwards. More than one role
    can match — pointing every role at one local model is the normal Ollama
    setup — and all matches are named, because all of them need editing.
    """
    from majordomo.config import MODEL_ROLES

    if not model:
        # An empty id matches every *unset* role, which would name
        # `agent_model` and `fallback_model` as the culprits for a request that
        # had nothing to do with either. "Not set" is not "set to this".
        return []

    return [
        (label, field, purpose)
        for label, field, purpose in MODEL_ROLES
        if (getattr(brain, field, "") or "") == model
    ]


def _explain(message: str, brain: BrainConfig, model: str, status: int | None) -> str:
    """Append what to do about it, when there is something to do.

    The failure this project is most likely to hand someone is a default going
    stale: the shipped models are OpenRouter *free* ids, and free ids appear and
    vanish. That arrived as a wall of JSON naming a model id, with nothing to say
    the id is a config value, which role was using it, or that ``mj config``
    exists — so the most probable failure read like a bug in the tool rather than
    a line to edit.

    Only permanent failures get advice. A 429 or a 503 has already been retried
    and has nothing to do with configuration, and telling someone to edit their
    models because a provider was briefly busy would be worse than saying
    nothing.
    """
    if status is None or status in RETRYABLE_STATUS:
        # No status means a transport failure; a retryable one has been retried.
        return message

    if status == 401:
        if not brain.api_key_env:
            # A keyless config meeting a 401 is not a missing key — it is a
            # disagreement. The config says this endpoint wants no auth and the
            # server says otherwise, so "check your key variable" would name a
            # variable that deliberately does not exist.
            return (
                f"{message}\n"
                f"  This endpoint wants credentials, but `brain.api_key_env` is "
                f"empty, which means\n"
                f"  \"no key needed\" — so none was sent. Either point it at a "
                f"variable holding a\n"
                f"  key, or check that `brain.base_url` is the local server you "
                f"meant."
            )
        return (
            f"{message}\n"
            f"  The provider rejected the credentials. Check {brain.api_key_env} in "
            f"~/.majordomo/.env.\n"
            f"  (`mj config` shows which variable is in use, and whether it is set.)"
        )

    lines = [
        f"  {model} was rejected and will keep being rejected — either the id or",
        "  this account's access to it. Free model ids come and go, so this is "
        "usually a",
        "  shipped default that has aged out rather than anything you did.",
    ]

    roles = _roles_using(brain, model)
    if roles:
        # One per line rather than run together in a sentence: more than one role
        # can point at the same model, and a comma-joined list both reads badly
        # and hides that each one is a separate line to edit.
        width = max(len(field) for _l, field, _p in roles)
        lines.append("  Set by:")
        lines += [f"    brain.{field:<{width}}   {purpose}" for _l, field, purpose in roles]

    # No path named here. `brain` does not carry where it was loaded from, so
    # printing the default location told anyone using `--config other.yml` to edit
    # the wrong file — confidently. `mj config` already prints which file is in
    # use, so pointing at it is both shorter and correct in every case.
    lines.append("  `mj config` shows all six roles and which file they came from.")
    return message + "\n" + "\n".join(lines)


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
            # Enriched here rather than where it was raised, so the advice is
            # composed exactly once however the request gave up.
            raise LLMError(
                _explain(str(primary), brain, model, primary.status), primary.status
            ) from primary
        try:
            return _request_one(messages, brain, fallback, extra)
        except QuotaExhausted:
            # The fallback hitting an account-wide cap is the same clear sentence
            # the primary's would have been. Wrapping it in "X failed; fallback
            # also failed" buried it.
            raise
        except LLMError as secondary:
            combined = (
                f"{model} failed ({primary}); fallback {fallback} also failed "
                f"({secondary})"
            )
            raise LLMError(
                _explain(combined, brain, model, primary.status), primary.status
            ) from secondary


def _request_one(
    messages: list[dict], brain: BrainConfig, model: str, extra: dict
) -> dict:
    """One model, with retries. See ``_request`` for the fallback layer."""
    # An empty ``api_key_env`` means "this endpoint wants no key at all", which
    # is how you reach a local server — Ollama, llama.cpp, LM Studio. Every
    # model became a config value, and ``base_url`` could already point
    # anywhere, but this check made a keyless endpoint unreachable: there was no
    # way to say "no key" that did not read as "you forgot the key".
    #
    # Naming a variable is still a promise that it holds something, so a *named*
    # variable that is unset keeps raising. That is the case the message below
    # was written for, and the common one.
    if brain.api_key_env:
        api_key = os.environ.get(brain.api_key_env)
        if not api_key:
            raise MissingApiKey(
                f"Set the {brain.api_key_env} environment variable with your "
                f"{brain.provider} API key."
            )
    else:
        api_key = ""

    url = f"{brain.base_url}/chat/completions"
    headers = {
        "Accept": "application/json",
        # OpenRouter uses these for attribution on free-tier models.
        "HTTP-Referer": "https://github.com/Navaneethp007/majordomo",
        "X-Title": "Majordomo",
    }
    if api_key:
        # Sent only when there is one. A local server that ignores the header is
        # common, but one that rejects `Bearer ` with nothing after it is not
        # unheard of, and an empty credential is worse than no credential.
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": brain.temperature,
        "max_tokens": 4096,
        **extra,
    }

    last_exc: Exception | None = None
    # The status of the last HTTP response, so the caller can tell a permanent
    # failure from a retried one. None means it never got that far.
    last_status: int | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = _post(url, headers=headers, json=payload, timeout=brain.timeout)
            if not response.is_success:
                last_exc = Exception(f"HTTP {response.status_code}: {response.text[:300]}")
                last_status = response.status_code

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

                if not _worth_waiting(response):
                    # This limit lifts in minutes, not seconds, so retrying the
                    # same model here is waiting for nothing. The fallback is
                    # still worth a try — it is a different model, and only an
                    # account-wide cap (above) makes that pointless too.
                    break

                if attempt < MAX_ATTEMPTS - 1:
                    # This is the whole point. The old policy retried
                    # immediately, so against a `retry-after: 5` the second
                    # attempt was guaranteed to fail too — every 429 was
                    # effectively one attempt with a cosmetic second.
                    time.sleep(_retry_after(response, attempt))
                continue

            body = response.json()
            envelope = _error_envelope(body)
            if envelope is not None:
                message, code = envelope
                # Checked here as well as in the status branch, because the cap
                # can arrive either way. Reaching it only through the status
                # branch meant a free-tier cap wearing a 200 was retried six
                # times across two models and reported as raw JSON — instead of
                # the sentence naming the reset time.
                exhausted = _daily_limit(response)
                if exhausted:
                    raise QuotaExhausted(exhausted)

                last_exc = Exception(
                    f"HTTP {response.status_code} carrying an error: {message}"
                )
                # The envelope's own code, not the 200 it arrived wearing — a
                # failure wearing a success code is still that failure.
                last_status = code if isinstance(code, int) else response.status_code
                # A 200 whose body is an error is a failure wearing a success
                # code, so classify it the way the status *should* have been.
                # Retrying a permanent one only adds latency to a certain
                # failure — the same reasoning as the status check above.
                if code is not None and code not in RETRYABLE_STATUS:
                    break
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(min(2.0**attempt, MAX_BACKOFF_SECONDS))
                continue

            if not isinstance(body, dict) or "choices" not in body:
                # The one case the old comment here anticipated, and the one it
                # reported worst: `str(KeyError('choices'))` is `"'choices'"`,
                # which says nothing at all. The body is the only thing that
                # explains it, so the body goes in the message.
                raise ValueError(
                    f"HTTP {response.status_code} but no 'choices' in the "
                    f"response: {response.text[:300]}"
                )

            choice = body["choices"][0]
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

    raise LLMError(str(last_exc), last_status)
