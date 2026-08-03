# Majordomo — Design Spec

**Date:** 2026-08-03
**Status:** Design — awaiting review
**CLI name:** `majordomo` (alias `mj`)
**Author:** Navaneeth

> A personal "mission control" that greets you when you return to your machine (boot / wake / login) and hands you **one synthesized, human-language briefing** of everything across your world that actually needs you — pending GitHub reviews, Linear updates, unread mail, **and the live state of your coding sessions** (idle, blocked-needing-approval, ended). It speaks the briefing aloud, and lets you act — Resume a session, Approve/Deny a blocked one — from a single surface, without hunting through terminals and tabs.

A majordomo is the head of a household staff who manages everyone and briefs the master. That is exactly this app's job: run the workers, fuse their reports, tell you the one thing that matters.

---

## 1. What this is (and is not)

**Is:** a local, single-user orchestration + briefing layer. Reads your streams, reasons over them, and produces a consolidated voice/text briefing plus an actionable "needs-you" list.

**Is not:** a per-source notification relay. Desktop apps already ping you "PR merged" / "new ticket." Majordomo does the opposite — it *digests* those streams into one intelligent summary and only surfaces what requires a decision.

**Honest framing:** the pure "consolidate messages" job is cheap (a few API fetches + one summarize call) and would not, on its own, justify a multi-agent system. The parts that *earn* orchestration — and are the point of this project — are:
1. **Session-awareness** — tracking live coding sessions and routing their "I'm blocked, decide this" moments to you.
2. **The escalation gate (router)** — spend real multi-agent compute *only* when a source is too big, needs judgment, or needs an action taken.

This is a **vitamin, not a painkiller** — a genuinely useful convenience and an excellent vehicle for learning multi-agent orchestration hands-on. Scope is kept honest accordingly.

---

## 2. v1 scope — DEPTH, with one live escalation (option B)

Prove the *novel, hard* half end-to-end before adding breadth.

**In scope for v1:**
- **One external source: GitHub.** (Linear + inbox are trivial copies of a proven adapter afterward.)
- **Full session-awareness** for **Claude Code** (terminal + VS Code extension) via hooks.
- **The router**, wired with **one real agent-mode escalation** so the multi-agent fork actually runs and is observable (chosen demo: a "too many items → reducer agent" fork — applied to GitHub in v1, generalizes to Linear later).
- **Consolidated briefing**, text + **voice (ElevenLabs only)**.
- A **surface**: an expandable tray widget on Windows.
- **Trigger** on wake / boot / login.

**Explicitly deferred (v2+):** Linear adapter, inbox/Gmail adapter, Codex/Cursor adapters, agents that *take actions* on external sources (option "B-full" — act, not just report), Fish Audio TTS, multi-machine.

---

## 3. Two models layers (important — not one vendor)

There are **two distinct model layers**; ElevenLabs covers only the second.

| Layer | Job | Provider (v1) | Notes |
|---|---|---|---|
| **Brain (LLM)** | reads raw source data, reasons, writes the briefing | free/cheap model via **OpenRouter** | provider + model id per *role* in config |
| **Voice (TTS)** | speaks the briefing text aloud | **ElevenLabs only** | paid + cloud; reuse Voicelog's existing TTS abstraction so Fish/local is a one-adapter swap later |

So v1 uses **two vendors by necessity** — one to think, one to speak.

---

## 4. Architecture

```
   wake / boot / login
          │
          ▼
   ┌─────────────┐     spawns in parallel
   │  Trigger    │────────────────────────────────┐
   │ (tray app)  │                                 │
   └─────────────┘                                 ▼
                                      ┌───────────────────────────┐
                                      │      Worker agents         │
                                      │  (one per source, ||)      │
                                      │                            │
        state.jsonl  ◀── hooks ───────┤  • GitHub worker           │
        (session events)              │  • Local-Sessions worker   │
                                      │      (reads state.jsonl)   │
                                      └───────────┬────────────────┘
                                                  │ each passes through
                                                  ▼
                                        ┌───────────────────┐
                                        │   Router / gate   │  cheap path by default;
                                        │  (per source)     │  escalates on triggers
                                        └───────┬───────────┘
                                                │ (worker summaries + escalated agent outputs)
                                                ▼
                                        ┌───────────────────┐
                                        │  Coordinator      │  fuses everything into
                                        │  (fuser agent)    │  ONE briefing + needs-you list
                                        └───────┬───────────┘
                                                ▼
                                ┌───────────────────────────────┐
                                │  Surface (tray widget)        │
                                │  • text briefing              │
                                │  • ElevenLabs voice           │
                                │  • actions: Resume / Approve  │
                                └───────────────────────────────┘
```

### Components (each has one clear job, testable in isolation)

1. **Trigger / tray app** — resident Windows tray process. Fires the pipeline on wake/boot/login (Windows Task Scheduler events + a resident listener). Owns the widget UI. Knows nothing about sources.
2. **Worker agents** — one per source. Each *fetches* its raw data and *reasons about its own domain* ("of this, what actually needs Nav?"). Reasoning per-worker is what makes them agents, not scripts. v1 workers: **GitHub**, **Local-Sessions**.
3. **Router / escalation gate** — sits between each worker's raw fetch and the coordinator. Default = cheap one-shot summarize. Escalates that *one source* to a reducer/worker agent when a trigger fires (§5).
4. **Coordinator (fuser)** — takes all worker outputs (cheap summaries and/or escalated agent results) and fuses them into a single natural-language briefing + a structured `needs_you[]` list. One model call.
5. **Session adapter (hooks)** — Claude Code hooks that write session lifecycle events to `state.jsonl` (§6). The Local-Sessions worker only ever reads that file.
6. **Voice** — ElevenLabs TTS over Voicelog's abstraction; speaks the briefing.
7. **Config** — one file: model per role (OpenRouter), voice provider + key, source credentials, thresholds.

---

## 5. The router / escalation gate (the multi-agent fork — B)

Per-source, **not** global. Decision kept mostly deterministic so we don't spend a model call just to decide whether to use a model.

| Trigger | Detected by | Why cheap path fails | Escalation |
|---|---|---|---|
| **Size** | token/byte count of raw payload > threshold | won't fit one prompt / blows free-model context | spawn a **reducer agent** that reasons the domain down before the coordinator sees it |
| **Action** | item is actionable **and** user/rule requested an action | a summary can't *do* anything; needs plan→act→observe→maybe-ask loop | spawn a **worker agent** with tools *(v2 — deferred; gate designed-in)* |
| **Reasoning** | optional tiny classifier: "needs judgment beyond summarizing?" | one-shot flattens nuance | escalate that one source to an agent |

**v1 wires the Size trigger for real** (the observable fork): on a heavy morning, GitHub/other small sources take the cheap path while an over-threshold source escalates to a reducer agent that returns judgment, not a dump. The Action path is designed into the interface but stubbed until v2.

Flow per source:
```
fetch → [gate] → within threshold & no action → cheap summarize   (default)
                 over threshold / needs action → spawn agent for THIS source only
```

---

## 6. Session-awareness (Claude Code) — via hooks, no transcript parsing

**Never parse `~/.claude/.../transcript.jsonl`** (format unstable). Instead, register `command` hooks in Claude Code `settings.json`; each appends one line to our own `~/.majordomo/state.jsonl`. Hook JSON is a stable contract.

| Session state | Hook (matcher) | We record |
|---|---|---|
| Started | `SessionStart` | `session_id`, `cwd`, surface, `status: active`, `at` |
| Topic (what it's about) | `UserPromptSubmit` | latest `prompt` → session topic *(robust summary source — no transcript needed)* |
| **Idle — finished, waiting on you** | `Notification` / `idle_prompt` | `status: idle_awaiting_you` |
| **Blocked — needs your approval** | `Notification` / `permission_prompt` | `status: blocked` (pure alert; does **not** hijack the decision) |
| Ended | `SessionEnd` | `status: ended` |

Every payload carries `session_id`, `cwd`, `permission_mode` on stdin, so each event self-identifies its session/project. The **Local-Sessions worker** reads `state.jsonl` and reports which sessions are live / idle / blocked, with each one's topic.

Note: `PermissionRequest` is the hook that can *control* a decision, but for a read-only watcher we deliberately use `Notification/permission_prompt` (alert only) — we surface it to the human, we don't auto-decide.

**Adapter pattern:** other tools (Codex/Cursor) write into the *same* `state.jsonl` in the *same* event shape via whatever surface they expose. Claude Code first (richest hooks); others v2.

---

## 7. Resume (jump back to the exact session)

Each session record stores its **surface** so Resume picks the right method:

| Surface | Resume method |
|---|---|
| **VS Code extension** | open documented URI `vscode://anthropic.claude-code/open?session=<session_id>` — opens/focuses that exact conversation (focuses the tab if already open) |
| **Terminal** | `claude --resume <session_id>`, run from the session's `cwd` |

Caveats baked in: VS Code and terminal keep **separate** session histories, and session-ID lookup is scoped to the session's project dir — so we always store and use the recorded `cwd` + surface. We do **not** rely on the undocumented `CLAUDE_CODE_BRIDGE_SESSION_ID`; we use the `session_id` from hook payloads, which is stable.

**Approve / Deny** on a blocked session: v1 surfaces the blocked state and provides **Open** (jump there to decide in-context). Actually auto-approving from the widget (via `PermissionRequest` hook returning a decision) is a v2 consideration — it means the widget can change behavior, so it's gated behind the same care as action-mode.

---

## 8. Data flow (one wake cycle)

1. Wake fires the tray app.
2. Tray spawns workers in parallel (GitHub, Local-Sessions).
3. Each worker fetches raw data (GitHub API call; Local-Sessions reads `state.jsonl`).
4. Router evaluates each source: cheap summarize, or escalate that source to a reducer agent (Size trigger).
5. Coordinator fuses all worker outputs → `{ briefing_text, needs_you[] }`.
6. Surface renders the panel + speaks `briefing_text` (ElevenLabs).
7. User acts: Resume / Open / dismiss. Actions call §7 methods.

**Internal event / record shapes:**
- Session event (in `state.jsonl`): `{ session_id, surface: "vscode"|"terminal", cwd, status, topic?, at }`
- Coordinator output: `{ briefing_text, needs_you: [{ kind, title, detail, action?, source }] }`

---

## 9. Error handling & degradation

- **A source fails** (GitHub down, bad token) → its worker returns an error stub; coordinator briefs the rest and notes "GitHub unavailable." Never blocks the whole briefing.
- **LLM (brain) fails** → fall back to a raw, un-fused list per source (the Voicelog degradation pattern).
- **TTS fails** → show text only, warn once; never crash.
- **No hooks installed yet / empty `state.jsonl`** → session section simply absent.
- **Escalation agent errors** → fall back to that source's cheap summarize (or a truncated note), never drop the source silently — log what was truncated.

---

## 10. Testing approach

- **Workers:** mock the source (GitHub API fixtures; a fixture `state.jsonl`); assert the worker's structured output.
- **Router:** unit-test the gate decision — small payload → cheap; over-threshold → escalate. Deterministic, no model needed.
- **Coordinator:** feed fixed worker outputs, assert shape of `{ briefing_text, needs_you[] }` (mock the model; assert prompt assembly + parsing).
- **Session adapter:** feed sample hook JSON payloads to the hook script; assert the `state.jsonl` line written for each event type.
- **Resume:** assert the correct URI / CLI command is *constructed* per surface (don't actually launch in tests).
- Network, audio, and model calls mocked throughout; follow Voicelog's test-first discipline.

---

## 11. Model / config (sketch)

Single config file (per-role model, so the brain is swappable without touching code):
```
brain:
  provider: openrouter
  fuser_model:   <free/cheap model id>
  reducer_model: <free/cheap model id>     # escalation agent
voice:
  provider: elevenlabs
  voice_id:  <id>
sources:
  github: { token_env: MAJORDOMO_GH_TOKEN }
router:
  size_threshold_tokens: 8000
```

---

## 12. Open questions / decide later

- [ ] Exact widget tech: Windows tray via a small Electron/Tauri app, or a local web page opened on wake? (leaning tray app; web page acceptable for the scrappy first cut)
- [ ] Which specific free OpenRouter model(s) for fuser vs reducer — pick after a quick quality/latency test.
- [ ] Should the widget ever be able to **Approve/Deny** a blocked session directly (v2, `PermissionRequest` control), or always **Open**-to-decide?
- [ ] Streaks / "quiet since" phrasing — nice-to-have voice polish, not v1.
- [ ] Name availability: confirm `majordomo` / `mj` free enough on npm + domain before publishing.

---

## 13. Naming trail

Concept explored as a "Jarvis-like" personal orchestrator. Candidates considered: Reveille, Aubade, Muster, Herald, Foyer, Otto. **Chosen: Majordomo** (`mj`) — the head-of-household who runs the staff and briefs the master; the most literal translation of the Jarvis role. Sibling in spirit to the `Familiar` concept (both are companion/assistant creatures for the dev's workflow), but a separate project.
