"""Tests for the Gmail worker. IMAP fully mocked — no network, no real mailbox.

Two properties dominate this module, because both are one word away from being
wrong and both would be actively harmful:

1. **Reading a briefing must never mark your mail as read.** That needs
   ``select(readonly=True)`` *and* ``BODY.PEEK`` rather than ``BODY[]``.
2. **Gmail must never contribute a needs-you item.** Email actionability is a
   judgement, and the last time a model was allowed to judge it from prose the
   briefing asserted three obligations that did not exist.
"""
from __future__ import annotations

import imaplib
from unittest import mock

import pytest

from majordomo import config as config_module
from majordomo.workers import gmail
from majordomo.workers.gmail import GmailError, Message

CFG = config_module.build(config_module.DEFAULTS)
ENV = {"GMAIL_ADDRESS": "nav@example.com", "GMAIL_APP_PASSWORD": "abcd efgh ijkl mnop"}

HEADERS = (
    b"From: Priya Raman <priya@example.com>\r\n"
    b"Subject: Invoice rounding looks off on the March batch\r\n\r\n"
)
LIST_HEADERS = (
    b"From: Blue Tokai <hello@bluetokaicoffee.com>\r\n"
    b"Subject: Tap for the tale behind Monsoon Malabar\r\n"
    b"List-Id: <news.bluetokaicoffee.com>\r\n\r\n"
)
# A real subject, MIME-encoded, containing an emoji — exactly what crashed the
# console before safe_print existed.
EMOJI_HEADERS = (
    b"From: GitHub <noreply@github.com>\r\n"
    b"Subject: =?UTF-8?B?8J+kliBBdXRvbWF0ZWQgcnVuIGZpbmlzaGVk?=\r\n\r\n"
)


def fake_imap(fetch_parts=None, search_uids=b"1 2 3", login_error=None):
    """A stand-in IMAP4_SSL recording how it was called."""
    client = mock.MagicMock(name="IMAP4_SSL")
    calls = {"select": None, "fetch_spec": None}

    if login_error:
        client.login.side_effect = login_error

    def select(mailbox, readonly=False):
        calls["select"] = (mailbox, readonly)
        return "OK", [b"1"]

    def uid(command, *args):
        if command == "SEARCH":
            return "OK", [search_uids]
        if command == "FETCH":
            calls["fetch_spec"] = args[-1]
            return "OK", fetch_parts if fetch_parts is not None else []
        return "OK", [b""]

    client.select.side_effect = select
    client.uid.side_effect = uid
    return client, calls


def part(thrid: int, headers: bytes):
    return (b"1 (X-GM-THRID %d UID 5 BODY[HEADER.FIELDS ...]" % thrid, headers)


# ---------------------------------------------------------------------------
# The two safety properties
# ---------------------------------------------------------------------------

def test_inbox_is_opened_readonly(monkeypatch):
    """Without readonly, Gmail marks messages seen as we touch them."""
    client, calls = fake_imap()
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)

    gmail.connect(CFG.sources.gmail, "nav@example.com", "pw")

    assert calls["select"] == ("INBOX", True)


def test_fetch_uses_body_peek(monkeypatch):
    """BODY[...] sets the \\Seen flag; BODY.PEEK[...] does not. Getting this
    wrong would silently mark every listed email read on every wake."""
    client, calls = fake_imap(fetch_parts=[part(123, HEADERS)])

    gmail.fetch(client, [b"1"])

    assert "BODY.PEEK[" in calls["fetch_spec"]
    assert "BODY[" not in calls["fetch_spec"].replace("BODY.PEEK[", "")


def test_gmail_never_contributes_a_needs_you_item(monkeypatch):
    client, _ = fake_imap(fetch_parts=[part(1, HEADERS), part(2, LIST_HEADERS)])
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)

    report = gmail.run(CFG, env=ENV)

    assert report.ok
    assert report.items == [], "email actionability is a judgement, not a fact"
    assert len(report.context_items) == 2


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_mime_encoded_subject_is_decoded(monkeypatch):
    client, _ = fake_imap(fetch_parts=[part(9, EMOJI_HEADERS)])
    messages = gmail.fetch(client, [b"1"])

    assert messages[0].subject.endswith("Automated run finished")
    assert "=?UTF-8?" not in messages[0].subject


def test_sender_display_name_is_preferred():
    assert gmail.sender_name("Priya Raman <priya@example.com>") == "Priya Raman"
    assert gmail.sender_name('"Blue Tokai" <a@b.com>') == "Blue Tokai"
    assert gmail.sender_name("bare@example.com") == "bare"


def test_thread_id_becomes_a_gmail_permalink(monkeypatch):
    client, _ = fake_imap(fetch_parts=[part(1872942872652183768, HEADERS)])
    messages = gmail.fetch(client, [b"1"])

    assert messages[0].url == "https://mail.google.com/mail/u/0/#inbox/19fe079e910878d8"


def test_mailing_list_mail_is_flagged(monkeypatch):
    client, _ = fake_imap(fetch_parts=[part(1, HEADERS), part(2, LIST_HEADERS)])
    messages = gmail.fetch(client, [b"1", b"2"])

    assert [m.is_list for m in messages] == [False, True]


def test_a_message_without_a_thread_id_still_parses(monkeypatch):
    client, _ = fake_imap(fetch_parts=[(b"1 (UID 5 BODY[HEADER.FIELDS ...]", HEADERS)])
    messages = gmail.fetch(client, [b"1"])

    assert len(messages) == 1
    assert messages[0].url is None


def test_fetch_with_no_uids_makes_no_call():
    client = mock.MagicMock()
    assert gmail.fetch(client, []) == []
    client.uid.assert_not_called()


# ---------------------------------------------------------------------------
# The digest text
# ---------------------------------------------------------------------------

def test_digest_marks_everything_no_action_needed():
    text = gmail.to_text([Message("Priya", "Invoice rounding", False, None)])
    assert "NO ACTION NEEDED" in text
    assert "must not be described as something to do" in text


def test_digest_separates_direct_mail_from_bulk():
    text = gmail.to_text([
        Message("Priya", "Invoice rounding", False, None),
        Message("Blue Tokai", "Monsoon Malabar", True, None),
    ])
    assert "Direct or transactional (1)" in text
    assert "Newsletters and automated mail (1)" in text


def test_empty_inbox_says_so():
    assert "no new mail" in gmail.to_text([])


def test_truncation_is_stated_not_silent():
    text = gmail.to_text([Message("A", "s", False, None)], truncated=57)
    assert "57 more not listed" in text


# ---------------------------------------------------------------------------
# run() — the boundary that must never raise
# ---------------------------------------------------------------------------

def test_missing_credentials_returns_a_stub_naming_the_vars():
    report = gmail.run(CFG, env={})
    assert report.ok is False
    assert "GMAIL_ADDRESS" in report.error and "GMAIL_APP_PASSWORD" in report.error


def test_app_password_spaces_are_stripped(monkeypatch):
    """Google shows app passwords in groups of four. Pasted verbatim they would
    fail auth for a completely invisible reason."""
    _, password = gmail.resolve_credentials(CFG.sources.gmail, ENV)
    assert password == "abcdefghijklmnop"


def test_a_rejected_login_returns_a_stub_with_a_hint(monkeypatch):
    client, _ = fake_imap(login_error=imaplib.IMAP4.error("AUTHENTICATIONFAILED"))
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)

    report = gmail.run(CFG, env=ENV)

    assert report.ok is False
    assert "app password" in report.error
    assert "2-Step" in report.error


def test_an_unreachable_server_returns_a_stub(monkeypatch):
    def boom(*a, **k):
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr(gmail, "_FastConnectIMAP", boom)
    report = gmail.run(CFG, env=ENV)

    assert report.ok is False
    assert "could not reach" in report.error


def test_run_never_raises_on_a_surprise(monkeypatch):
    monkeypatch.setattr(gmail, "connect", mock.Mock(side_effect=RuntimeError("surprise")))
    report = gmail.run(CFG, env=ENV)

    assert report.ok is False
    assert "unexpected" in report.error


def test_the_connection_is_always_closed(monkeypatch):
    client, _ = fake_imap(fetch_parts=[part(1, HEADERS)])
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)

    gmail.run(CFG, env=ENV)

    client.logout.assert_called_once()


def test_max_messages_caps_a_huge_result(monkeypatch):
    """The account behind this has 18,103 unread. An unbounded query must not be
    able to turn into an 18k-message fetch."""
    many = b" ".join(str(i).encode() for i in range(500))
    client, calls = fake_imap(search_uids=many, fetch_parts=[part(1, HEADERS)])
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)

    report = gmail.run(CFG, env=ENV)

    fetched = calls["fetch_spec"]
    assert fetched is not None
    assert "480 more not listed" in report.summary


def test_report_is_pre_summarised(monkeypatch):
    """to_text already reads as prose; a summarise pass would spend a call and
    blur the sender/subject detail that makes it useful."""
    client, _ = fake_imap(fetch_parts=[part(1, HEADERS)])
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)

    assert gmail.run(CFG, env=ENV).pre_summarised is True


# ---------------------------------------------------------------------------
# Bulk detection — List-Id alone caught 3 of 18 real messages
# ---------------------------------------------------------------------------

import email as _email  # noqa: E402


def parsed(raw: bytes):
    return _email.message_from_bytes(raw)


def test_list_unsubscribe_marks_bulk():
    """The reliable signal. List-Id alone filed TLDR, Swiggy and Supabase under
    'from people' — wrong about 80% of a real inbox."""
    assert gmail.is_bulk(parsed(
        b"From: TLDR <n@tldr.tech>\r\nList-Unsubscribe: <https://x/u>\r\n\r\n"
    )) is True


def test_precedence_bulk_marks_bulk():
    assert gmail.is_bulk(parsed(b"From: a@b.com\r\nPrecedence: bulk\r\n\r\n")) is True


def test_auto_submitted_marks_bulk():
    assert gmail.is_bulk(parsed(b"From: a@b.com\r\nAuto-Submitted: auto-generated\r\n\r\n")) is True


def test_auto_submitted_no_is_not_bulk():
    assert gmail.is_bulk(parsed(b"From: a@b.com\r\nAuto-Submitted: no\r\n\r\n")) is False


def test_noreply_sender_marks_bulk():
    assert gmail.is_bulk(parsed(b"From: Foo <no-reply@foo.com>\r\n\r\n")) is True
    assert gmail.is_bulk(parsed(b"From: Foo <donotreply@foo.com>\r\n\r\n")) is True


def test_a_plain_message_is_not_bulk():
    assert gmail.is_bulk(parsed(
        b"From: Priya Raman <priya@example.com>\r\nSubject: Invoice\r\n\r\n"
    )) is False


def test_max_messages_zero_fetches_nothing(monkeypatch):
    """`uids[-0:]` is `uids[0:]`, so a cap of 0 turned the hard limit into no
    limit — and reported every message as truncated at the same time."""
    client, calls = fake_imap(search_uids=b"1 2 3", fetch_parts=[])
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)
    data = config_module._deep_merge(
        config_module.DEFAULTS, {"sources": {"gmail": {"max_messages": 0}}}
    )

    report = gmail.run(config_module.build(data), env=ENV)

    assert calls["fetch_spec"] is None, "a cap of 0 must fetch nothing at all"
    assert report.context_items == []


def test_missing_credentials_are_flagged_unconfigured():
    """A setup state, not an outage — the speaker treats the two differently."""
    report = gmail.run(CFG, env={})
    assert report.ok is False
    assert report.unconfigured is True


def test_a_real_failure_is_not_flagged_unconfigured(monkeypatch):
    client, _ = fake_imap(login_error=imaplib.IMAP4.error("AUTHENTICATIONFAILED"))
    monkeypatch.setattr(gmail, "_FastConnectIMAP", lambda *a, **k: client)

    report = gmail.run(CFG, env=ENV)

    assert report.ok is False
    assert report.unconfigured is False, "a rejected password is a real outage"
