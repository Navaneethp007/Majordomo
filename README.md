# Majordomo

> A majordomo is the head of a household staff who manages everyone and briefs the master.

One synthesized, spoken briefing of everything across your world that actually needs you —
pending GitHub reviews and the live state of your coding sessions (idle, blocked awaiting
approval, ended) — delivered when you come back to your machine.

Majordomo is **not** a notification relay. Your apps already ping you "PR merged" / "new
ticket." Majordomo does the opposite: it *digests* those streams into one intelligent summary
and surfaces only what requires a decision.

```
mj brief          # fetch, route, fuse, print and speak the briefing
mj sessions       # list live / idle / blocked coding sessions
mj resume <id>    # jump back into the exact session, on the right surface
mj install-hooks  # wire session-awareness into Claude Code
```

Status: **in development.** See
[the design spec](docs/superpowers/specs/2026-08-03-majordomo-design.md).

## Install

```
pip install -e .[dev]
```

Requires Python 3.10+. Configuration lives at `~/.majordomo/config.yml` and is entirely
optional — every setting has a default.

## License

MIT
