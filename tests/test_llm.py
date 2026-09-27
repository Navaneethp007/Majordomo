"""Tests for the brain client. All HTTP mocked."""
from __future__ import annotations

import time
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


# ---------------------------------------------------------------------------
# Keyless endpoints
#
# `base_url` was always configurable, so the docs offered an Ollama config —
# but every request demanded a key first, so that config could never have run.
# An empty `api_key_env` is the spelling for "this endpoint wants no key".
# ---------------------------------------------------------------------------

KEYLESS = replace(BRAIN, api_key_env="", base_url="http://localhost:11434/v1")


def test_an_empty_key_env_sends_no_authorization_header(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = {}

    def capture(url, headers=None, **kwargs):
        seen["url"] = url
        seen["headers"] = headers
        return response({"choices": [{"message": {"content": "local"}}]})

    monkeypatch.setattr(llm, "_post", capture)

    assert llm.complete(MSG, KEYLESS, "llama3.2") == "local"
    # Absent, not empty: a server that rejects `Bearer ` with nothing after it
    # is worse than one that never saw the header.
    assert "Authorization" not in seen["headers"]
    assert seen["url"].startswith("http://localhost:11434/v1")


def test_mj_config_reports_a_keyless_setup_as_needing_no_key(monkeypatch, capsys):
    """Not "NOT SET": there is no variable, and nothing is missing."""
    from majordomo import cli, paths
    import yaml

    paths.ensure_home()
    paths.default_config_path().write_text(
        yaml.safe_dump({"brain": {"api_key_env": "", "provider": "ollama"}}),
        encoding="utf-8",
    )

    cli.main(["config"])
    out = capsys.readouterr().out

    assert "none needed" in out
    assert "NOT SET" not in out


def test_a_named_but_unset_variable_still_raises(monkeypatch):
    """Naming a variable is a promise that it holds something."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingApiKey):
        llm.complete(MSG, BRAIN, "some/model")


# ---------------------------------------------------------------------------
# Naming the knob on a permanent failure
#
# The failure this project is most likely to hand someone is a shipped default
# going stale: the defaults are OpenRouter *free* ids, and free ids appear and
# vanish. That arrived as a wall of JSON naming a model id, with nothing to say
# the id is a config value, which role used it, or that `mj config` exists — so
# the most probable failure read like a bug in the tool.
# ---------------------------------------------------------------------------

def _responder(status, body, monkeypatch):
    def post(url, headers=None, **_kwargs):
        r = mock.MagicMock()
        r.is_success = 200 <= status < 300
        r.status_code = status
        r.text = body
        r.headers = {}
        import json

        r.json.return_value = json.loads(body)
        return r

    monkeypatch.setattr(llm, "_post", post)


def test_a_vanished_model_names_the_role_and_the_file(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    _responder(404, '{"error":{"message":"No endpoints found","code":404}}', monkeypatch)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, BRAIN, BRAIN.chat_model)

    text = str(exc.value)
    assert "brain.chat_model" in text
    assert "mj ask, mj chat" in text   # the role's purpose, not just its name
    assert "mj config" in text         # how to find the file and the other roles


def test_every_role_using_the_model_is_named(monkeypatch):
    """Pointing all six roles at one model is the normal local setup, and each
    one is a separate line to change."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    brain = replace(
        BRAIN, worker_model="one/model", chat_model="one/model", fallback_model=""
    )
    _responder(404, '{"error":{"message":"gone","code":404}}', monkeypatch)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, brain, "one/model")

    text = str(exc.value)
    assert "brain.worker_model" in text
    assert "brain.chat_model" in text


def test_a_transient_failure_gets_no_config_advice(monkeypatch):
    """A busy provider has nothing to do with your models, and telling someone to
    edit them because a pool was briefly full is worse than saying nothing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    _responder(503, '{"error":{"message":"busy"}}', monkeypatch)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, NO_FALLBACK, "some/model")

    assert "mj config" not in str(exc.value)
    assert "brain." not in str(exc.value)


def test_a_transport_failure_gets_no_config_advice(monkeypatch):
    """No status at all — a DNS or TLS failure says nothing about configuration."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")

    def boom(*_args, **_kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(llm, "_post", boom)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, NO_FALLBACK, "some/model")

    assert "mj config" not in str(exc.value)


def test_a_rejected_key_points_at_the_env_var_not_the_models(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-wrong")
    _responder(401, '{"error":{"message":"No auth credentials","code":401}}', monkeypatch)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, NO_FALLBACK, "some/model")

    text = str(exc.value)
    assert "OPENROUTER_API_KEY" in text
    assert ".env" in text
    assert "brain.chat_model" not in text  # not a model problem


def test_a_401_on_a_keyless_config_does_not_name_a_blank_variable(monkeypatch):
    """A keyless 401 is a disagreement, not a missing key.

    The config says this endpoint wants no auth and the server says otherwise, so
    "check  in ~/.majordomo/.env" — with nothing where the variable should be —
    was both ungrammatical and the wrong advice. Reachable via LM Studio with auth
    turned on, or a hosted base_url left behind when api_key_env was blanked.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    brain = replace(BRAIN, api_key_env="", fallback_model="")
    _responder(401, '{"error":{"message":"No auth credentials","code":401}}', monkeypatch)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, brain, brain.chat_model)

    text = str(exc.value)
    assert "Check  in" not in text            # the blank
    assert "brain.api_key_env` is empty" in text
    assert "base_url" in text                  # the other thing it could be


def test_the_advice_names_no_config_path(monkeypatch):
    """`brain` does not carry where it was loaded from, so printing the default
    location told anyone using `--config other.yml` to edit the wrong file."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    _responder(404, '{"error":{"message":"gone","code":404}}', monkeypatch)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, replace(BRAIN, fallback_model=""), BRAIN.chat_model)

    text = str(exc.value)
    assert "config.yml" not in text
    assert "mj config" in text  # which does know, and prints it


def test_a_permanent_error_wearing_a_200_still_gets_advice(monkeypatch):
    """A failure in a 200's body is still that failure — it must classify by the
    envelope's own code, not the status it arrived wearing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    _responder(200, '{"error":{"message":"No endpoints found","code":404}}', monkeypatch)

    with pytest.raises(LLMError) as exc:
        llm.complete(MSG, replace(BRAIN, fallback_model=""), BRAIN.chat_model)

    assert "brain.chat_model" in str(exc.value)


def test_the_role_table_is_the_one_in_config():
    """Read backwards from `brain`, so there is no second copy to drift."""
    from majordomo.config import MODEL_ROLES

    fields = {field for _label, field, _purpose in MODEL_ROLES}
    assert fields <= set(BRAIN.__dataclass_fields__)

    # The shipped defaults point `fuser` and `fallback` at the same id, so both
    # match — which is the behaviour wanted, since both are lines to edit.
    assert [f for _l, f, _p in llm._roles_using(BRAIN, BRAIN.fuser_model)] == [
        "fuser_model",
        "fallback_model",
    ]
    assert llm._roles_using(BRAIN, "nothing/points/here") == []

    # An unset role must not match every model that happens to be "".
    blank = replace(BRAIN, agent_model="", fallback_model="")
    assert llm._roles_using(blank, "") == []


def test_a_daily_cap_on_the_fallback_keeps_its_own_sentence(monkeypatch):
    """It was wrapped in "X failed; fallback also failed", which buried the one
    message that says when it comes back."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    calls = []

    def post(url, headers=None, json=None, **_kwargs):
        calls.append(json["model"])
        r = mock.MagicMock()
        r.headers = {}
        if len(calls) == 1:
            r.is_success = False
            r.status_code = 404
            r.text = '{"error":{"message":"gone","code":404}}'
            r.json.return_value = {"error": {"message": "gone", "code": 404}}
            return r
        r.is_success = False
        r.status_code = 429
        r.text = '{"error":{"message":"free-models-per-day limit reached"}}'
        r.json.return_value = {"error": {"message": "free-models-per-day"}}
        return r

    monkeypatch.setattr(llm, "_post", post)

    with pytest.raises(llm.QuotaExhausted):
        llm.complete(MSG, BRAIN, BRAIN.chat_model)


def test_a_keyless_config_ignores_a_key_that_happens_to_be_set(monkeypatch):
    """`api_key_env: ""` means no key, not "find one somewhere"."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    seen = {}

    def capture(url, headers=None, **kwargs):
        seen["headers"] = headers
        return response({"choices": [{"message": {"content": "local"}}]})

    monkeypatch.setattr(llm, "_post", capture)

    llm.complete(MSG, KEYLESS, "llama3.2")
    assert "Authorization" not in seen["headers"]


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




def response_of(status, body, headers=None):
    return httpx.Response(
        status_code=status, text=body, headers=headers or {},
        request=httpx.Request("POST", "https://example.test"),
    )


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


# ---------------------------------------------------------------------------
# A 200 that is really a failure
# ---------------------------------------------------------------------------


ENVELOPE = '{"error": {"message": "upstream provider returned nothing", "code": 502}}'
PERMANENT = '{"error": {"message": "no such model", "code": 404}}'


def test_an_error_envelope_names_the_error_not_the_exception(monkeypatch):
    """`str(KeyError('choices'))` is `"'choices'"` — the least informative
    possible account of a failure whose cause was sitting in the body."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    with mock.patch.object(llm, "_post", return_value=response_of(200, ENVELOPE)):
        with mock.patch.object(llm.time, "sleep", lambda *_: None):
            with pytest.raises(llm.LLMError) as exc_info:
                llm.complete(MSG, NO_FALLBACK, "m")

    message = str(exc_info.value)
    assert "upstream provider returned nothing" in message
    assert message != "'choices'"


def test_a_permanent_envelope_is_not_retried(monkeypatch):
    """A 200 carrying a 404 is a failure wearing a success code — classify it
    the way the status should have been."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = mock.Mock(return_value=response_of(200, PERMANENT))

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.LLMError, match="no such model"):
            llm.complete(MSG, NO_FALLBACK, "m")

    assert post.call_count == 1


def test_a_retryable_envelope_is_retried(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = mock.Mock(return_value=response_of(200, ENVELOPE))

    with mock.patch.object(llm, "_post", post):
        with mock.patch.object(llm.time, "sleep", lambda *_: None):
            with pytest.raises(llm.LLMError):
                llm.complete(MSG, NO_FALLBACK, "m")

    assert post.call_count == llm.MAX_ATTEMPTS


def test_a_body_of_the_wrong_shape_shows_the_body(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = '{"id": "x", "object": "chat.completion"}'

    with mock.patch.object(llm, "_post", return_value=response_of(200, body)):
        with mock.patch.object(llm.time, "sleep", lambda *_: None):
            with pytest.raises(llm.LLMError) as exc_info:
                llm.complete(MSG, NO_FALLBACK, "m")

    assert "chat.completion" in str(exc_info.value)


def test_an_ordinary_completion_is_unaffected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    good = '{"choices": [{"message": {"content": "hello"}}]}'

    with mock.patch.object(llm, "_post", return_value=response_of(200, good)):
        assert llm.complete(MSG, NO_FALLBACK, "m") == "hello"


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"error": {"message": "boom", "code": 502}}, ("boom", 502)),
        ({"error": "plain string"}, ("plain string", None)),
        ({"error": {"message": "no code"}}, ("no code", None)),
        ({"choices": []}, None),
        ("not a dict", None),
    ],
)
def test_the_envelope_reader_handles_what_providers_actually_send(body, expected):
    assert llm._error_envelope(body) == expected


# ---------------------------------------------------------------------------
# Codes that are not integers, and caps that are not 429s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        (404, 404),
        ("404", 404),
        ("model_not_found", 404),
        ("invalid_api_key", 401),
        ("RATE_LIMIT_EXCEEDED", 429),
        ("something_nobody_has_seen", None),
        (None, None),
        ({"nested": 1}, None),
    ],
)
def test_error_codes_are_read_whatever_shape_they_arrive_in(code, expected):
    """OpenAI-compatible bodies commonly send string codes. Dropping them left
    `code` as None, which skipped the non-retryable break — so a bad key spent
    all six attempts proving it was still a bad key."""
    assert llm._as_status(code) == expected


def test_a_string_permanent_code_is_not_retried(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = '{"error": {"message": "no such model", "code": "model_not_found"}}'
    post = mock.Mock(return_value=response_of(200, body))

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.LLMError, match="no such model"):
            llm.complete(MSG, NO_FALLBACK, "m")

    assert post.call_count == 1


def test_an_unknown_string_code_is_retried(monkeypatch):
    """Wrong in this direction costs seconds; wrong the other way turns a
    transient failure into a permanent one."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = '{"error": {"message": "who knows", "code": "brand_new_thing"}}'
    post = mock.Mock(return_value=response_of(200, body))

    with mock.patch.object(llm, "_post", post):
        with mock.patch.object(llm.time, "sleep", lambda *_: None):
            with pytest.raises(llm.LLMError):
                llm.complete(MSG, NO_FALLBACK, "m")

    assert post.call_count == llm.MAX_ATTEMPTS




def test_an_ordinary_200_envelope_is_still_just_an_error(monkeypatch):
    """The cap check must not swallow every envelope."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = '{"error": {"message": "upstream died", "code": 502}}'

    with mock.patch.object(llm, "_post", return_value=response_of(200, body)):
        with mock.patch.object(llm.time, "sleep", lambda *_: None):
            with pytest.raises(llm.LLMError) as exc_info:
                llm.complete(MSG, NO_FALLBACK, "m")

    assert not isinstance(exc_info.value, llm.QuotaExhausted)
    assert "upstream died" in str(exc_info.value)


# ---------------------------------------------------------------------------
# An error key is not, by itself, an error
# ---------------------------------------------------------------------------


GOOD_CHOICES = [{"message": {"content": "a good answer"}}]


@pytest.mark.parametrize("error", [{}, "", 0, [], None])
def test_an_empty_error_key_is_not_an_error(error):
    """`{}`, `""` and `0` all became an "error" whose message was their repr."""
    assert llm._error_envelope({"choices": GOOD_CHOICES, "error": error}) is None
    assert llm._error_envelope({"error": error}) is None


def test_a_usable_completion_beats_an_error_key_beside_it():
    """Some providers send `error: {}` alongside real choices. Treating the
    key's presence as the signal discarded the answer, retried three times,
    burned the fallback and raised."""
    body = {"choices": GOOD_CHOICES, "error": {"message": "something odd"}}
    assert llm._error_envelope(body) is None


def test_the_completion_still_comes_back(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = '{"choices": [{"message": {"content": "hello"}}], "error": {}}'
    post = mock.Mock(return_value=response_of(200, body))

    with mock.patch.object(llm, "_post", post):
        assert llm.complete(MSG, NO_FALLBACK, "m") == "hello"

    assert post.call_count == 1          # no retry, no fallback


def test_a_real_envelope_with_no_choices_is_still_caught():
    body = {"error": {"message": "upstream died", "code": 502}}
    assert llm._error_envelope(body) == ("upstream died", 502)


def test_empty_choices_do_not_mask_a_real_error():
    """`choices: []` is not a usable completion."""
    body = {"choices": [], "error": {"message": "upstream died", "code": 502}}
    assert llm._error_envelope(body) == ("upstream died", 502)


# ---------------------------------------------------------------------------
# A quota is a reset you cannot wait out, whatever it is called
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Two questions about a 429, and they have different answers
# ---------------------------------------------------------------------------
#
#   will waiting help?      the reset distance — structural, survives a rename
#   will another model help? whose limit it is — only the provider's wording says
#
# Deciding both from the distance skipped a fallback that would have worked.
# Deciding both from the wording sent us back to six attempts on a spent quota.


def stamp_in(seconds: float) -> str:
    """A reset stamp that distance away, in milliseconds since the epoch.

    Relative on purpose. A hardcoded stamp made these tests pass until the date
    it named and quietly change meaning afterwards — which is exactly what
    happened to the first version of them.
    """
    return str(int((time.time() + seconds) * 1000))


def limited(message: str = "busy", reset_in: float | None = None, headers=None):
    body = '{"error":{"message":"%s"' % message
    if reset_in is not None:
        body += ',"metadata":{"headers":{"X-RateLimit-Reset":"%s"}}' % stamp_in(reset_in)
    body += "}}"
    return response_of(429, body, headers or {})


ACCOUNT_CAP = "Rate limit exceeded: free-models-per-day"


# --- will waiting help? ----------------------------------------------------


def test_a_reset_minutes_away_is_not_worth_waiting_for():
    assert llm._worth_waiting(limited(reset_in=6 * 3600)) is False


def test_a_reset_seconds_away_is():
    assert llm._worth_waiting(limited(reset_in=5)) is True


def test_a_stamp_already_in_the_past_means_the_limit_lifted():
    assert llm._worth_waiting(limited(reset_in=-3600)) is True


def test_silence_is_not_evidence_of_a_quota():
    """Refusing to retry on no information is the worse mistake."""
    assert llm._worth_waiting(response_of(429, '{"error":{"message":"busy"}}')) is True


def test_an_unparseable_stamp_is_treated_as_silence():
    body = '{"error":{"metadata":{"headers":{"X-RateLimit-Reset":"tomorrow"}}}}'
    assert llm._worth_waiting(response_of(429, body)) is True


def test_a_real_retry_after_header_beats_the_body():
    """A header is a protocol answer; the body is a provider's JSON."""
    response = limited(reset_in=6 * 3600, headers={"retry-after": "5"})
    assert llm._worth_waiting(response) is True


@pytest.mark.parametrize(
    "key", ["X-RateLimit-Reset", "x-ratelimit-reset", "X-Ratelimit-Reset"]
)
def test_the_stamp_is_read_whatever_its_casing(key):
    """The casing of a key inside a JSON body is the provider's whim."""
    body = '{"error":{"metadata":{"headers":{"%s":"%s"}}}}' % (key, stamp_in(6 * 3600))
    assert llm._worth_waiting(response_of(429, body)) is False


# --- will another model help? ----------------------------------------------


def test_an_account_wide_cap_is_named_and_dated():
    message = llm._daily_limit(limited(ACCOUNT_CAP, reset_in=6 * 3600))

    assert "Out of free requests" in message
    assert "clears at" in message


def test_a_rename_that_still_reads_as_daily_is_caught():
    """Several spellings exist for the one condition, so the list is broader
    than the two OpenRouter happens to use today."""
    assert llm._daily_limit(limited("you have hit your daily limit", reset_in=6 * 3600))


def test_one_models_pool_is_not_an_account_cap():
    """A model busy for ten minutes is a good reason to try the fallback."""
    assert llm._daily_limit(limited("provider busy", headers={"retry-after": "600"})) == ""


def test_a_near_reset_is_never_an_account_cap():
    assert llm._daily_limit(limited(ACCOUNT_CAP, reset_in=5)) == ""


# --- what each one does to the request ------------------------------------


def test_an_account_cap_costs_one_attempt_and_no_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = mock.Mock(return_value=limited(ACCOUNT_CAP, reset_in=6 * 3600))
    capped = replace(BRAIN, fallback_model="some/other-model:free")

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.QuotaExhausted) as exc_info:
            llm.complete(MSG, capped, "m")

    assert post.call_count == 1
    assert "Out of free requests" in str(exc_info.value)


def test_a_long_pool_limit_stops_retrying_but_still_falls_back(monkeypatch):
    """The case that split this in two: deciding from the reset distance alone
    skipped the fallback that would have worked."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = mock.Mock(return_value=limited("provider busy", headers={"retry-after": "600"}))
    with_fallback = replace(BRAIN, fallback_model="some/other-model:free")

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.LLMError):
            llm.complete(MSG, with_fallback, "m")

    assert post.call_count == 2          # one each, neither retried


def test_a_busy_pool_still_retries_and_falls_back(monkeypatch):
    """The behaviour none of this may break."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    post = mock.Mock(return_value=limited("busy", headers={"retry-after": "0"}))
    with_fallback = replace(BRAIN, fallback_model="some/other-model:free")

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.LLMError):
            llm.complete(MSG, with_fallback, "m")

    assert post.call_count == llm.MAX_ATTEMPTS * 2


def test_an_account_cap_wearing_a_200_is_caught_too(monkeypatch):
    """OpenRouter answers 200 with an error body when an upstream dies, and the
    cap can arrive that way."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = (
        '{"error":{"message":"%s","metadata":{"headers":{"X-RateLimit-Reset":"%s"}}}}'
        % (ACCOUNT_CAP, stamp_in(6 * 3600))
    )
    post = mock.Mock(return_value=response_of(200, body))

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.QuotaExhausted):
            llm.complete(MSG, replace(BRAIN, fallback_model="other:free"), "m")

    assert post.call_count == 1


def test_quota_exhausted_still_degrades_like_any_llm_error():
    """An LLMError subclass, so every existing caller keeps degrading rather
    than crashing on a new exception type."""
    assert issubclass(llm.QuotaExhausted, llm.LLMError)


def test_an_account_cap_with_no_reset_information_is_still_a_cap():
    """The regression: `_worth_waiting` returns True on missing information —
    right for retrying, wrong as a gate here. A body saying `free-models-per-day`
    is a daily cap whether or not anyone said when it lifts."""
    body = (
        '{"error":{"message":"Rate limit exceeded: free-models-per-day. '
        'Add 10 credits to unlock 1000 free model requests per day","code":429}}'
    )
    message = llm._daily_limit(response_of(429, body))

    assert "Out of free requests" in message
    assert "clears at" not in message          # nothing said when, so it does not claim


def test_a_daily_wording_on_something_clearing_in_seconds_is_not_a_cap():
    """A provider being loose with words, not a spent quota. Only a *known*
    short distance overrides the marker."""
    response = response_of(429, '{"error":{"message":"daily limit"}}', {"retry-after": "5"})
    assert llm._daily_limit(response) == ""


def test_the_cap_with_no_reset_costs_one_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    body = '{"error":{"message":"free-models-per-day","code":429}}'
    post = mock.Mock(return_value=response_of(429, body))
    capped = replace(BRAIN, fallback_model="other/model:free")

    with mock.patch.object(llm, "_post", post):
        with pytest.raises(llm.QuotaExhausted):
            llm.complete(MSG, capped, "m")

    assert post.call_count == 1
