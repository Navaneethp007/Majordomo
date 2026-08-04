"""The local panel — the briefing as a page, with the actions attached.

A localhost HTTP server rather than a native window: it is a few hundred lines
instead of a second runtime, and the buttons that matter (Open, Resume) are just
links.

Two things are deliberate. It binds to **127.0.0.1 only**, and every request
carries a per-run **token**. Without the token any page you happened to have
open in a browser could POST to ``/api/resume`` and start launching editors —
localhost is not a trust boundary on a shared machine.

v1 surfaces a blocked session and takes you there to decide in context. It does
not approve or deny on your behalf; that would mean this page can change what
your agent is allowed to do, which is gated behind the same care as action-mode
(spec §7).
"""
from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from majordomo import brief, resume as resume_mod, state
from majordomo.config import Config
from majordomo.workers.sessions import describe, fold

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Majordomo</title>
<style>
  :root { color-scheme: light dark; --fg:#1a1a1a; --muted:#6b6b6b; --bg:#fbfbfa;
          --card:#fff; --line:#e6e4e0; --urgent:#b4432c; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#ececec; --muted:#9a9a9a; --bg:#191918; --card:#232322;
            --line:#333331; --urgent:#e08265; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:2.5rem 1.5rem; background:var(--bg); color:var(--fg);
         font:16px/1.6 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
  main { max-width: 42rem; margin: 0 auto; }
  h1 { font-size:.75rem; text-transform:uppercase; letter-spacing:.12em;
       color:var(--muted); font-weight:600; margin:0 0 1.25rem; }
  .briefing { font-size:1.25rem; line-height:1.5; margin:0 0 2.5rem;
              text-wrap: pretty; }
  h2 { font-size:.75rem; text-transform:uppercase; letter-spacing:.12em;
       color:var(--muted); font-weight:600; margin:0 0 .75rem; }
  .item { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:.9rem 1.1rem; margin-bottom:.6rem; display:flex; gap:1rem;
          align-items:center; justify-content:space-between; }
  .item.blocked { border-left:3px solid var(--urgent); }
  .title { font-weight:500; }
  .detail { color:var(--muted); font-size:.875rem; }
  button, a.btn { font:inherit; font-size:.875rem; padding:.4rem .9rem; cursor:pointer;
          border-radius:7px; border:1px solid var(--line); background:transparent;
          color:var(--fg); text-decoration:none; white-space:nowrap; }
  button:hover { border-color:var(--muted); }
  .empty { color:var(--muted); }
  footer { margin-top:2.5rem; color:var(--muted); font-size:.8125rem; }
</style>
<main>
  <h1>Majordomo</h1>
  <p class="briefing" id="briefing">Loading…</p>
  <h2 id="needs-heading" hidden>Needs you</h2>
  <div id="items"></div>
  <footer><button onclick="load(true)">Re-brief</button></footer>
</main>
<script>
const TOKEN = new URLSearchParams(location.search).get('t');

async function load(fresh) {
  const el = document.getElementById('briefing');
  if (fresh) el.textContent = 'Briefing…';
  const res = await fetch(`/api/brief?t=${TOKEN}`);
  const data = await res.json();
  el.textContent = data.briefing_text;

  const items = document.getElementById('items');
  document.getElementById('needs-heading').hidden = data.needs_you.length === 0;
  items.innerHTML = '';
  for (const item of data.needs_you) {
    const row = document.createElement('div');
    row.className = 'item' + (item.kind === 'session_blocked' ? ' blocked' : '');
    const left = document.createElement('div');
    left.innerHTML = `<div class="title"></div><div class="detail"></div>`;
    left.querySelector('.title').textContent = item.title;
    left.querySelector('.detail').textContent = item.detail;
    row.append(left);

    if (item.action && item.source === 'sessions') {
      const b = document.createElement('button');
      b.textContent = 'Resume';
      b.onclick = async () => {
        b.textContent = 'Opening…';
        await fetch(`/api/resume?t=${TOKEN}&id=${encodeURIComponent(item.action)}`,
                    {method: 'POST'});
        b.textContent = 'Resume';
      };
      row.append(b);
    } else if (item.action) {
      const a = document.createElement('a');
      a.className = 'btn'; a.href = item.action; a.target = '_blank';
      a.textContent = 'Open';
      row.append(a);
    }
  }
}
load(false);
</script>
"""


@dataclass
class PanelHandle:
    url: str
    server: HTTPServer
    thread: threading.Thread

    def stop(self) -> None:
        self.server.shutdown()


def _make_handler(config: Config, token: str):
    class Handler(BaseHTTPRequestHandler):
        # Silence the default stderr access log — this runs behind a tray icon.
        def log_message(self, *args):  # noqa: A003
            pass

        def _authed(self, query: dict) -> bool:
            supplied = (query.get("t") or [""])[0]
            # Constant-time: the token is the only thing standing between a
            # stray browser tab and launching processes.
            return secrets.compare_digest(supplied, token)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if not self._authed(query):
                self._json(403, {"error": "bad token"})
                return

            if parsed.path == "/api/brief":
                result = brief.run(config)
                self._json(
                    200,
                    {
                        "briefing_text": result.briefing.briefing_text,
                        "needs_you": [
                            {
                                "kind": i.kind,
                                "title": i.title,
                                "detail": i.detail,
                                "source": i.source,
                                "action": i.action,
                            }
                            for i in result.briefing.needs_you
                        ],
                    },
                )
                return

            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if not self._authed(query):
                self._json(403, {"error": "bad token"})
                return

            if parsed.path == "/api/resume":
                session_id = (query.get("id") or [""])[0]
                live = fold(
                    state.read_events(),
                    stale_after_hours=config.sources.sessions.stale_after_hours,
                )
                match = next((s for s in live if s.session_id == session_id), None)
                if match is None:
                    self._json(404, {"error": "no such live session"})
                    return
                try:
                    resume_mod.launch(resume_mod.build(match))
                except resume_mod.ResumeError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                self._json(200, {"ok": True, "opened": describe(match)})
                return

            self._json(404, {"error": "not found"})

    return Handler


def serve(config: Config, port: int = 0) -> PanelHandle:
    """Start the panel on localhost. ``port=0`` picks a free one."""
    token = secrets.token_urlsafe(24)
    server = HTTPServer(("127.0.0.1", port), _make_handler(config, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    actual = server.server_address[1]
    return PanelHandle(url=f"http://127.0.0.1:{actual}/?t={token}", server=server, thread=thread)
