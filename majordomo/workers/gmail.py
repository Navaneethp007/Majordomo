"""The Gmail worker — a digest, never a to-do list.

This source produces **zero** ``needs_you`` items, on purpose.

GitHub and sessions work because actionability is a *fact*: a review was
formally requested; a session is stopped at a permission prompt. Email has no
equivalent. "Needs a reply" is a judgement, and the one time this project let a
model judge actionability from prose it invented three obligations that did not
exist. So Gmail reports what is there and says nothing about what you must do.

Two implementation choices are load-bearing for safety:

``select(..., readonly=True)`` and ``BODY.PEEK[...]`` rather than ``BODY[...]``.
Together they guarantee that reading a briefing can never mark your mail as
read. Both are one word away from being wrong, and a tool that silently marked
seventeen emails read on every wake would be worse than useless.

The third choice is about scale. This account has 18,103 unread; ``to:me``
narrows that to 17,882, because it matches anything delivered to the address,
mailing lists included. Only recency plus Gmail's own Primary category cuts it
to something briefable (~17/day), which is why the default query has both and
why ``max_messages`` caps it regardless of what the query is edited to.
"""
from __future__ import annotations

import email
import imaplib
import re
import socket
from dataclasses import dataclass
from email.header import decode_header, make_header

from majordomo.config import Config, GmailConfig
from majordomo.models import ContextItem, SourceReport

NAME = "gmail"

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993

#: Gmail's thread id is a 64-bit integer whose *hex* is the id the web UI uses,
#: which makes a real permalink out of an IMAP fetch.
THREAD_URL = "https://mail.google.com/mail/u/0/#inbox/{thrid:x}"

_THRID = re.compile(rb"X-GM-THRID (\d+)")


class GmailError(Exception):
    """A Gmail fetch failed. Caught at the worker boundary, never escapes."""


class GmailNotConfigured(GmailError):
    """No credentials have been supplied. A setup state, not an outage."""


@dataclass(frozen=True)
class Message:
    sender: str
    subject: str
    is_list: bool
    url: str | None


def _decode(raw: str | None, fallback: str = "") -> str:
    """Decode a MIME-encoded header. Real subjects arrive as ``=?UTF-8?B?…?=``."""
    if not raw:
        return fallback
    try:
        return str(make_header(decode_header(raw))).strip() or fallback
    except (UnicodeDecodeError, LookupError, ValueError):
        # A malformed header is not worth losing the message over.
        return raw.strip() or fallback


def is_bulk(parsed) -> bool:
    """Is this machine-sent mail rather than a person writing to you?

    ``List-Id`` alone is not enough: of 18 real messages it caught 3, leaving
    TLDR, Swiggy, Supabase and a hackathon newsletter filed under "from people".
    A digest whose entire value is "did a human write to me" cannot be wrong
    about that 80% of the time.

    ``List-Unsubscribe`` is the reliable signal — bulk senders include it almost
    universally, and unsubscribe links are effectively mandatory for marketing
    mail in most jurisdictions. ``Precedence: bulk`` and ``Auto-Submitted`` catch
    the automated stragglers, and a no-reply sender is self-describing.
    """
    if parsed.get("List-Id") or parsed.get("List-Unsubscribe"):
        return True
    if (parsed.get("Precedence") or "").strip().lower() in ("bulk", "list", "junk"):
        return True
    auto = (parsed.get("Auto-Submitted") or "").strip().lower()
    if auto and auto != "no":
        return True
    sender = (parsed.get("From") or "").lower()
    return bool(re.search(r"(no[-_.]?reply|donotreply|do[-_.]?not[-_.]?reply)@", sender))


def sender_name(raw: str) -> str:
    """'Nav <nav@example.com>' -> 'Nav'; bare addresses keep their local part."""
    decoded = _decode(raw, "(unknown sender)")
    match = re.match(r"\s*(.+?)\s*<[^>]+>\s*$", decoded)
    if match:
        return match.group(1).strip().strip('"') or decoded
    if "@" in decoded:
        return decoded.split("@")[0].strip("<> ")
    return decoded


class _FastConnectIMAP(imaplib.IMAP4_SSL):
    """IMAP4_SSL that doesn't stall for a minute on a broken IPv6 route.

    ``imap.gmail.com`` resolves to two IPv6 addresses *before* its IPv4 ones. On
    a machine with no working IPv6 — which this one is — ``create_connection``
    walks that list in order and waits out a full TCP timeout on each before
    reaching an address that works. Measured here: **42 seconds** to connect,
    against 0.1s connecting to the IPv4 address directly.

    So: try every resolved address with a short per-address budget, IPv4 first.
    A working IPv6 host still connects on the first attempt; a broken one costs
    ``connect_timeout`` instead of the OS default. TLS still validates against
    ``self.host``, so preferring an address family changes nothing about trust.
    """

    def __init__(self, host, port, connect_timeout: float, **kwargs):
        self._connect_timeout = connect_timeout
        super().__init__(host, port, **kwargs)

    def _create_socket(self, timeout=None):
        infos = socket.getaddrinfo(self.host, self.port, 0, socket.SOCK_STREAM)
        # IPv4 first. Not a judgement about IPv6 — just that on a desktop the
        # cost of guessing wrong is asymmetric: a wasted 0.1s versus 40 seconds.
        infos.sort(key=lambda info: info[0] != socket.AF_INET)

        last: Exception | None = None
        for family, socktype, proto, _canon, sockaddr in infos:
            sock = socket.socket(family, socktype, proto)
            try:
                sock.settimeout(self._connect_timeout)
                sock.connect(sockaddr)
                # Restore the operational timeout for reads and writes.
                sock.settimeout(timeout)
                return self.ssl_context.wrap_socket(sock, server_hostname=self.host)
            except OSError as exc:
                last = exc
                sock.close()

        raise last or OSError(f"could not connect to {self.host}:{self.port}")


def connect(cfg: GmailConfig, address: str, password: str) -> imaplib.IMAP4_SSL:
    """Open a read-only INBOX. Raises GmailError; the caller makes a stub."""
    try:
        client = _FastConnectIMAP(
            IMAP_HOST, IMAP_PORT, connect_timeout=cfg.connect_timeout, timeout=cfg.timeout
        )
    except OSError as exc:
        raise GmailError(f"could not reach {IMAP_HOST}: {exc}") from exc

    try:
        client.login(address, password)
    except imaplib.IMAP4.error as exc:
        raise GmailError(
            f"Gmail rejected the login — check {cfg.password_env} is a current "
            f"app password and 2-Step Verification is still on ({exc})"
        ) from exc

    # readonly is the difference between reading your mail and *marking it read*.
    typ, _ = client.select("INBOX", readonly=True)
    if typ != "OK":
        raise GmailError("could not open INBOX")
    return client


def search(client: imaplib.IMAP4_SSL, query: str) -> list[bytes]:
    """UIDs matching a Gmail-syntax query, newest last."""
    try:
        typ, data = client.uid("SEARCH", None, f'X-GM-RAW "{query}"')
    except imaplib.IMAP4.error as exc:
        raise GmailError(f"search failed: {exc}") from exc
    if typ != "OK":
        raise GmailError(f"search rejected: {query!r}")
    return data[0].split() if data and data[0] else []


def fetch(client: imaplib.IMAP4_SSL, uids: list[bytes]) -> list[Message]:
    """Headers plus thread id for each uid. BODY.PEEK never sets the Seen flag."""
    if not uids:
        return []

    try:
        typ, response = client.uid(
            "FETCH",
            b",".join(uids),
            "(X-GM-THRID BODY.PEEK[HEADER.FIELDS "
            "(FROM SUBJECT LIST-ID LIST-UNSUBSCRIBE PRECEDENCE AUTO-SUBMITTED)])",
        )
    except imaplib.IMAP4.error as exc:
        raise GmailError(f"fetch failed: {exc}") from exc
    if typ != "OK":
        raise GmailError("fetch rejected")

    messages: list[Message] = []
    for part in response or []:
        if not isinstance(part, tuple) or len(part) < 2:
            continue
        prefix, body = part[0], part[1]

        thrid = _THRID.search(prefix or b"")
        url = THREAD_URL.format(thrid=int(thrid.group(1))) if thrid else None

        try:
            parsed = email.message_from_bytes(body)
        except Exception:
            continue

        messages.append(
            Message(
                sender=sender_name(parsed.get("From", "")),
                subject=_decode(parsed.get("Subject"), "(no subject)"),
                is_list=is_bulk(parsed),
                url=url,
            )
        )

    return messages


def to_text(messages: list[Message], truncated: int = 0) -> str:
    """The digest. Every line marked NO ACTION NEEDED, because none of it is.

    The marker is not decoration — the fuser is instructed to respect it, and it
    is what stops "seventeen unread" becoming "you must read seventeen emails".
    """
    if not messages:
        return "NO ACTION NEEDED — no new mail in your primary inbox."

    people = [m for m in messages if not m.is_list]
    bulk = [m for m in messages if m.is_list]

    lines = [
        f"NO ACTION NEEDED — {len(messages)} unread in the primary inbox. This is "
        "background: none of it is a request awaiting a decision, and it must not "
        "be described as something to do.",
    ]
    if people:
        # Deliberately not "from people". The signal detects bulk mail reliably;
        # it says nothing about humanness. What is left is direct and
        # transactional mail — password resets, security alerts, a real reply —
        # and labelling that "from people" would overclaim what we actually know.
        lines.append(f"  Direct or transactional ({len(people)}):")
        lines += [f"    - {m.sender}: {m.subject}" for m in people]
    if bulk:
        lines.append(f"  Newsletters and automated mail ({len(bulk)}):")
        lines += [f"    - {m.sender}: {m.subject}" for m in bulk]
    if truncated:
        lines.append(f"  ({truncated} more not listed.)")

    return "\n".join(lines)


def resolve_credentials(cfg: GmailConfig, env: dict[str, str] | None = None) -> tuple[str, str]:
    import os

    source = env if env is not None else os.environ
    address = (source.get(cfg.address_env) or "").strip()
    # Google displays app passwords in groups of four; the spaces are cosmetic
    # and pasting them verbatim would otherwise fail auth for an invisible reason.
    password = (source.get(cfg.password_env) or "").replace(" ", "").strip()

    if not address or not password:
        raise GmailNotConfigured(
            f"no Gmail credentials — set {cfg.address_env} and {cfg.password_env} "
            f"in ~/.majordomo/.env"
        )
    return address, password


def run(config: Config, env: dict[str, str] | None = None) -> SourceReport:
    """Fetch and report. Never raises — a failure becomes an error stub."""
    cfg = config.sources.gmail
    client = None
    try:
        address, password = resolve_credentials(cfg, env)
        client = connect(cfg, address, password)

        uids = search(client, cfg.query)
        # Clamp before slicing: `uids[-0:]` is `uids[0:]`, so a cap of 0 turned
        # the hard limit into no limit at all — and then reported every message
        # as truncated at the same time.
        cap = max(0, cfg.max_messages)
        truncated = max(0, len(uids) - cap)
        # Newest last in IMAP, so the tail is the most recent.
        messages = fetch(client, uids[-cap:] if cap else [])
    except GmailNotConfigured as exc:
        return SourceReport.failed(NAME, str(exc), unconfigured=True)
    except GmailError as exc:
        return SourceReport.failed(NAME, str(exc))
    except Exception as exc:
        return SourceReport.failed(NAME, f"unexpected: {exc}")
    finally:
        if client is not None:
            try:
                client.logout()
            except Exception:
                pass

    return SourceReport(
        source=NAME,
        ok=True,
        summary=to_text(messages, truncated),
        items=[],  # deliberately empty; see the module docstring
        context_items=[
            ContextItem(
                kind="unread_mail",
                title=f"{m.sender}: {m.subject}",
                detail="Newsletter or bulk mail." if m.is_list else "Direct or transactional.",
                source=NAME,
                action=m.url,
            )
            for m in messages
        ],
        path="cheap",
        route_reason="local digest, no model needed",
        # to_text already reads as prose for a human; a summarise pass would
        # spend a call and blur the sender/subject detail that makes it useful.
        pre_summarised=True,
    )
