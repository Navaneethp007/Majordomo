"""Tests for the brain client. All HTTP mocked."""
from __future__ import annotations

from unittest import mock

import httpx
import pytest

from majordomo import config as config_module, llm
from majordomo.llm import LLMError, MissingApiKey

BRAIN = config_module.build(config_module.DEFAULTS).brain
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
    with mock.patch("majordomo.llm.httpx.post", return_value=response(payload)):
        assert llm.complete(MSG, BRAIN, "m") == "the answer"


def test_null_content_becomes_empty_string(monkeypatch):
    """Free-tier models answer `content: null` on a filtered or empty
    completion. Returned verbatim it produced an AttributeError in callers that
    only guard against LLMError — normalising here covers all three at once."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    payload = {"choices": [{"message": {"content": None}}]}
    with mock.patch("majordomo.llm.httpx.post", return_value=response(payload)):
        assert llm.complete(MSG, BRAIN, "m") == ""


def test_non_string_content_becomes_empty_string(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    payload = {"choices": [{"message": {"content": [{"type": "text"}]}}]}
    with mock.patch("majordomo.llm.httpx.post", return_value=response(payload)):
        assert llm.complete(MSG, BRAIN, "m") == ""


def test_retries_once_then_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    with mock.patch("majordomo.llm.httpx.post", return_value=response({}, 500)) as post:
        with pytest.raises(LLMError):
            llm.complete(MSG, BRAIN, "m")
    assert post.call_count == 2


def test_unexpected_body_shape_is_llmerror(monkeypatch):
    """A proxy error envelope returned as HTTP 200 must degrade, not crash."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    with mock.patch("majordomo.llm.httpx.post", return_value=response({"error": "nope"})):
        with pytest.raises(LLMError):
            llm.complete(MSG, BRAIN, "m")


def test_network_failure_is_llmerror(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    with mock.patch("majordomo.llm.httpx.post", side_effect=httpx.ConnectError("down")):
        with pytest.raises(LLMError):
            llm.complete(MSG, BRAIN, "m")


def test_role_model_is_sent_not_a_config_field(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    payload = {"choices": [{"message": {"content": "x"}}]}
    with mock.patch("majordomo.llm.httpx.post", return_value=response(payload)) as post:
        llm.complete(MSG, BRAIN, "vendor/reducer-model")
    assert post.call_args.kwargs["json"]["model"] == "vendor/reducer-model"
