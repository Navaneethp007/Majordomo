"""Tests for the brain client. All HTTP mocked."""
from __future__ import annotations

from dataclasses import replace
from unittest import mock

import httpx
import pytest

from majordomo import config as config_module, llm
from majordomo.llm import LLMError, MissingApiKey

BRAIN = config_module.build(config_module.DEFAULTS).brain

#: The retry tests care about one model in isolation. Turning the fallback
#: off is a supported configuration, so they exercise the real entry point
#: rather than a wrapper that existed only for them.
NO_FALLBACK = replace(BRAIN, fallback_model="")
MSG = [{"role": "user", "content": "hi"}]


def response(payload, status=200):
    r = mock.MagicMock()
    r.is_success = 200 <= status < 300
    r.status_code = status
    r.text = "body"
    r.json.return_value = payload
    return r


def test_missing_key_raises_and_names_the_env_var(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingApiKey) as exc:
        llm.complete(MSG, BRAIN, "some/model")
    assert "OPENROUTER_API_KEY" in str(exc.value)


def test_happy_path_returns_content(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    payload = {"choices": [{"message": {"content": "the answer"}}]}
    with mock.patch("majordomo.llm._post", return_value=response(payload)):
        assert llm.complete(MSG, BRAIN, "m") == "the answer"


def test_null_content_becomes_empty_string(monkeypatch):
    """Free-tier models answer `content: null` on a filtered or empty
    completion. Returned verbatim it produced an AttributeError in callers that
    only guard against LLMError — normalising here covers all three at once."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    payload = {"choices": [{"message": {"content": None}}]}
    with mock.patch("majordomo.llm._post", return_value=response(payload)):
        assert llm.complete(MSG, BRAIN, "m") == ""


def test_non_string_content_becomes_empty_string(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    payload = {"choices": [{"message": {"content": [{"type": "text"}]}}]}
    with mock.patch("majordomo.llm._post", return_value=response(payload)):
        assert llm.complete(MSG, BRAIN, "m") == ""


def test_a_server_error_is_retried_then_raises(monkeypatch):
    """Asserts the count comes from MAX_ATTEMPTS rather than a literal.

    This used to pin `== 2`, from a policy that retried immediately. Against a
    provider that answers `retry-after: 5` that second attempt was guaranteed to
    fail, so the number was describing a bug rather than an intent.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    with mock.patch("majordomo.llm._post", return_value=response({}, 500)) as post:
        with pytest.raises(LLMError):
            llm.complete(MSG, NO_FALLBACK, "m")
    assert post.call_count == llm.MAX_ATTEMPTS


def test_unexpected_body_shape_is_llmerror(monkeypatch):
    """A proxy error envelope returned as HTTP 200 must degrade, not crash."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    with mock.patch("majordomo.llm._post", return_value=response({"error": "nope"})):
        with pytest.raises(LLMError):
            llm.complete(MSG, BRAIN, "m")


def test_network_failure_is_llmerror(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    with mock.patch("majordomo.llm._post", side_effect=httpx.ConnectError("down")):
        with pytest.raises(LLMError):
            llm.complete(MSG, BRAIN, "m")


def test_role_model_is_sent_not_a_config_field(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    payload = {"choices": [{"message": {"content": "x"}}]}
    with mock.patch("majordomo.llm._post", return_value=response(payload)) as post:
        llm.complete(MSG, BRAIN, "vendor/reducer-model")
    assert post.call_args.kwargs["json"]["model"] == "vendor/reducer-model"


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

def failing(status, headers=None):
    r = mock.MagicMock()
    r.is_success = False
    r.status_code = status
    r.text = "body"
    r.headers = headers or {}
    return r


def ok():
    r = response({"choices": [{"message": {"content": "ok"}}]})
    r.headers = {}
    return r


def test_a_429_waits_for_retry_after_then_succeeds(monkeypatch):
    """The whole point. Retrying instantly against `retry-after: 5` is the
    same as not retrying — which is what every 429 did before."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    with mock.patch.object(
        llm, "_post", side_effect=[failing(429, {"retry-after": "5"}), ok()]
    ):
        assert llm.complete(MSG, BRAIN, "m") == "ok"

    assert slept == [5.0]


def test_an_absurd_retry_after_is_capped(monkeypatch):
    """A scheduled task that waits ten minutes looks hung."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    with mock.patch.object(
        llm, "_post", side_effect=[failing(429, {"retry-after": "600"}), ok()]
    ):
        llm.complete(MSG, BRAIN, "m")

    assert slept == [llm.MAX_BACKOFF_SECONDS]


def test_a_non_numeric_retry_after_falls_back_to_backoff(monkeypatch):
    """`retry-after` may legally be an HTTP date; we do not parse those."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    with mock.patch.object(
        llm,
        "_post",
        side_effect=[failing(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), ok()],
    ):
        llm.complete(MSG, BRAIN, "m")

    assert slept == [1.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_failures_are_not_retried(monkeypatch, status):
    """A 403 means the model is gated to partner apps; a 404 means the id is
    wrong. Retrying only adds latency to a certain failure."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    post = mock.Mock(side_effect=[failing(status)] * 5)

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(LLMError) as exc:
            llm.complete(MSG, NO_FALLBACK, "m")

    assert post.call_count == 1
    assert slept == []
    assert str(status) in str(exc.value)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
def test_retryable_failures_use_every_attempt(monkeypatch, status):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    post = mock.Mock(side_effect=[failing(status)] * llm.MAX_ATTEMPTS)

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(LLMError):
            llm.complete(MSG, NO_FALLBACK, "m")

    assert post.call_count == llm.MAX_ATTEMPTS


def test_backoff_grows_and_never_exceeds_the_cap(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    with mock.patch.object(llm, "_post", side_effect=[failing(500)] * llm.MAX_ATTEMPTS):
        with pytest.raises(LLMError):
            llm.complete(MSG, NO_FALLBACK, "m")

    assert slept == sorted(slept)                       # monotonic
    assert all(s <= llm.MAX_BACKOFF_SECONDS for s in slept)
    assert len(slept) == llm.MAX_ATTEMPTS - 1           # no sleep after the last


def test_a_transport_error_is_retried_with_backoff(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    with mock.patch.object(
        llm, "_post", side_effect=[httpx.ConnectError("down"), ok()]
    ):
        assert llm.complete(MSG, BRAIN, "m") == "ok"

    assert slept == [1.0]


def test_success_on_the_first_try_never_sleeps(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    with mock.patch.object(llm, "_post", return_value=ok()):
        assert llm.complete(MSG, BRAIN, "m") == "ok"

    assert slept == []


# ---------------------------------------------------------------------------
# The fallback layer
# ---------------------------------------------------------------------------

def brain_with(primary_fallback: str):
    return config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS, {"brain": {"fallback_model": primary_fallback}}
        )
    ).brain


def test_a_busy_primary_falls_back_to_the_reachable_model(monkeypatch):
    """The two axes pull apart: the model that follows the fuse prompt best sits
    behind the busiest free pool. Measured — GLM 429'd through every attempt
    even honouring retry-after, while minimax answered."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    brain = brain_with("vendor/reachable")

    post = mock.Mock(side_effect=[failing(429)] * llm.MAX_ATTEMPTS + [ok()])
    with mock.patch.object(llm, "_post", post):
        assert llm.complete(MSG, brain, "vendor/busy") == "ok"

    models = [c.kwargs["json"]["model"] for c in post.call_args_list]
    assert models[:llm.MAX_ATTEMPTS] == ["vendor/busy"] * llm.MAX_ATTEMPTS
    assert models[-1] == "vendor/reachable"


def test_a_permanent_failure_still_falls_back(monkeypatch):
    """A 403 is permanent for *that model* — inkling is gated to partner apps —
    but says nothing about the fallback."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    brain = brain_with("vendor/reachable")

    post = mock.Mock(side_effect=[failing(403), ok()])
    with mock.patch.object(llm, "_post", post):
        assert llm.complete(MSG, brain, "vendor/gated") == "ok"

    assert post.call_count == 2          # no retry on the 403, straight to fallback


def test_both_failing_names_both_in_the_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    brain = brain_with("vendor/reachable")

    with mock.patch.object(llm, "_post", return_value=failing(429)):
        with pytest.raises(LLMError) as exc:
            llm.complete(MSG, brain, "vendor/busy")

    assert "vendor/busy" in str(exc.value)
    assert "vendor/reachable" in str(exc.value)


def test_a_fallback_equal_to_the_primary_is_skipped(monkeypatch):
    """Otherwise the same busy pool is asked twice for no reason."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    brain = brain_with("vendor/same")

    post = mock.Mock(return_value=failing(429))
    with mock.patch.object(llm, "_post", post):
        with pytest.raises(LLMError):
            llm.complete(MSG, brain, "vendor/same")

    assert post.call_count == llm.MAX_ATTEMPTS


def test_an_empty_fallback_disables_the_layer(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    brain = brain_with("")

    post = mock.Mock(return_value=failing(429))
    with mock.patch.object(llm, "_post", post):
        with pytest.raises(LLMError):
            llm.complete(MSG, brain, "vendor/busy")

    assert post.call_count == llm.MAX_ATTEMPTS


def test_a_working_primary_never_touches_the_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = mock.Mock(return_value=ok())
    with mock.patch.object(llm, "_post", post):
        llm.complete(MSG, brain_with("vendor/reachable"), "vendor/primary")

    assert post.call_count == 1
    assert post.call_args.kwargs["json"]["model"] == "vendor/primary"


def test_a_missing_key_is_not_retried_on_the_fallback(monkeypatch):
    """MissingApiKey is unrecoverable and identical for every model."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with mock.patch.object(llm, "_post") as post:
        with pytest.raises(MissingApiKey):
            llm.complete(MSG, brain_with("vendor/reachable"), "vendor/primary")
    assert not post.called


# ---------------------------------------------------------------------------
# The two kinds of 429
# ---------------------------------------------------------------------------


DAILY_BODY = (
    '{"error":{"message":"Rate limit exceeded: free-models-per-day. Add 10 '
    'credits to unlock 1000 free model requests per day","code":429,'
    '"metadata":{"headers":{"X-RateLimit-Limit":"50","X-RateLimit-Remaining":"0",'
    '"X-RateLimit-Reset":"1788825600000"},"limit_source":"openrouter_free_tier_daily"}}}'
)
POOL_BODY = (
    '{"error":{"message":"Provider busy","code":429,'
    '"metadata":{"limit_source":"upstream_provider_shared_pool"}}}'
)


def response_of(status, body, headers=None):
    return httpx.Response(
        status_code=status, text=body, headers=headers or {},
        request=httpx.Request("POST", "https://example.test"),
    )


def test_the_daily_cap_is_recognised_and_dated():
    """'Try again later' is not actionable; 'back at 05:30' is."""
    message = llm._daily_limit(response_of(429, DAILY_BODY))

    assert "Out of free requests" in message
    assert "resets at" in message


def test_a_busy_pool_is_not_a_daily_cap():
    """These arrive as the same status and mean opposite things: one clears in
    five seconds, the other at midnight."""
    assert llm._daily_limit(response_of(429, POOL_BODY)) == ""


def test_the_daily_cap_is_not_retried(monkeypatch):
    """Retrying spends more requests against a limit already reached."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    post = mock.Mock(return_value=response_of(429, DAILY_BODY))

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.QuotaExhausted):
            llm.complete(MSG, NO_FALLBACK, "m")

    assert post.call_count == 1


def test_the_daily_cap_skips_the_fallback_too(monkeypatch):
    """The cap is on the account, so the fallback is exactly as exhausted."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    post = mock.Mock(return_value=response_of(429, DAILY_BODY))
    capped = replace(BRAIN, fallback_model="some/other-model:free")

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.QuotaExhausted) as exc_info:
            llm.complete(MSG, capped, "m")

    assert post.call_count == 1
    assert "also failed" not in str(exc_info.value)


def test_a_busy_pool_still_retries_and_falls_back(monkeypatch):
    """The behaviour this must not break."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    post = mock.Mock(return_value=response_of(429, POOL_BODY, {"retry-after": "0"}))
    with_fallback = replace(BRAIN, fallback_model="some/other-model:free")

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.LLMError):
            llm.complete(MSG, with_fallback, "m")

    assert post.call_count == llm.MAX_ATTEMPTS * 2       # both models, fully retried


def test_quota_exhausted_still_degrades_like_any_llm_error():
    """An LLMError subclass, so every existing caller keeps degrading rather
    than crashing on a new exception type."""
    assert issubclass(llm.QuotaExhausted, llm.LLMError)


def test_an_unparseable_reset_still_names_the_problem():
    body = DAILY_BODY.replace('"1788825600000"', '"not-a-number"')
    message = llm._daily_limit(response_of(429, body))

    assert "Out of free requests" in message
    assert "resets at" not in message


# ---------------------------------------------------------------------------
# The pooled client
# ---------------------------------------------------------------------------


def test_the_client_is_created_once_and_reused(monkeypatch):
    """httpx.post opens a fresh connection per call — a TCP handshake and a TLS
    negotiation each time, measured at ~167ms. Invisible for a briefing's four
    requests; the agent makes up to 24 in a row for one task."""
    monkeypatch.setattr(llm, "_client", None)
    made = []

    class FakeClient:
        def __init__(self, **kwargs):
            made.append(kwargs)

        def post(self, url, headers=None, json=None, timeout=None):
            return response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm.httpx, "Client", FakeClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    for _ in range(5):
        llm.complete(MSG, NO_FALLBACK, "m")

    assert len(made) == 1


def test_the_post_seam_carries_the_configured_timeout(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    seen = {}

    def spy(url, headers, json, timeout):
        seen["timeout"] = timeout
        return response({"choices": [{"message": {"content": "ok"}}]})

    with mock.patch("majordomo.llm._post", spy):
        llm.complete(MSG, NO_FALLBACK, "m")

    assert seen["timeout"] == BRAIN.timeout
