# Majordomo

> A majordomo is the head of a household staff who manages everyone and briefs the master.

A personal assistant that runs on your machine, knows what you have been building, and
does something about it. You talk to it, it answers with your context already loaded, and
it can act — read code, write files, run tests — asking before anything changes.

It is **not** a notification relay, and it is **not** a daemon. Your apps already ping you
"PR merged" / "new ticket". Majordomo does the opposite: it digests those streams and
surfaces only what needs a decision, when you ask it to.

```
mj chat                      # a conversation, with your memory and activity loaded
mj ask "what did I ship?"    # one question, same context, no session
mj do "add tests for X"      # the agent works in this directory, asking before it writes
mj brief                     # one spoken summary of what actually needs you
```

Status: **in development**, used daily by its author. Windows first; nothing is
deliberately platform-locked but nothing else is tested.

## Install

```
pip install -e .[dev]
```

Python 3.10+. Configuration is entirely optional — every setting has a default:

```
mj config           what is in force right now, and which model does what
mj config --init    write an annotated ~/.majordomo/config.yml to edit
```

API keys go in `~/.majordomo/.env`. The config file names environment *variables*, never
values, so it stays safe to commit.

### Choosing your own models

Majordomo talks to one OpenAI-compatible `/chat/completions` endpoint, so **any** provider
speaking that shape works — OpenRouter, OpenAI, Groq, Together, vLLM, or Ollama on your own
machine. Point `brain.base_url` and `brain.api_key_env` at it:

```yaml
brain:
  provider: ollama
  base_url: http://localhost:11434/v1
  api_key_env: OLLAMA_API_KEY     # unused; Ollama ignores auth
  chat_model: qwen2.5:14b
  agent_model: qwen2.5-coder:14b
  fallback_model: ""              # nothing to fall back to
```

Anthropic's own API is not OpenAI-shaped, so reach Claude models through OpenRouter
(`anthropic/claude-sonnet-4.5`) rather than pointing `base_url` at it.

There are **six model roles**, because they want genuinely different things and one model
is rarely best at all of them:

| Role | Job | Pick it for |
|---|---|---|
| `worker_model` | compress one source's raw payload | speed; the job is mechanical |
| `fuser_model` | write the four spoken sentences | **instruction-following.** It must never turn context into a task |
| `reducer_model` | handle a payload too big for the worker | a long context window |
| `chat_model` | `mj ask`, `mj chat` | consistency — you are waiting on it, so a long tail hurts more than a slow median |
| `agent_model` | `mj do`, `/agent` | tool calling and code. Empty reuses `chat_model` |
| `fallback_model` | tried once when a role's model fails | being on a *different* provider from the rest |

`fuser_model` is the one worth testing properly. It receives a list of things needing you
and a list of things that don't, and must say "nothing needs you" when the first list is
empty. A model that promotes something from the second list makes the whole tool
untrustworthy — that is the single check to run against any candidate.

[majordomo/config.example.yml](majordomo/config.example.yml) explains every setting.

Optional extras, none required for text:

| Extra | For |
|---|---|
| `.[nvidia]` | speech, in and out, via NVIDIA Riva (gRPC, hence separate) |
| `.[voice]` | microphone capture — the one dependency speech *input* costs |
| `.[tray]` | the resident tray icon |

## Commands

### Talking to it

```
mj chat                   an interactive session with your context loaded
  --resume [ID]           continue the most recent conversation, or one by id
  --list                  list saved conversations
mj ask <question>         one question, answered with the same context
```

Inside `mj chat`:

| | |
|---|---|
| `/context` | what memory and activity are loaded, and how large it has grown |
| `/clear` | start fresh, keeping the loaded context |
| `/remember` | propose what from this conversation is worth keeping |
| `/agent TASK` | put the agent to work without leaving the conversation |
| `/build NAME` | scaffold a repo from this conversation |
| `/voice` | speak your next message instead of typing it |

### Doing things

```
mj do <task>              the built-in agent works in the current directory
  --directory, -C DIR     work somewhere else
  --yes                   approve every write and command without asking
mj review <path>          open Claude Code on a repo with /code-review
mj start <idea>           scaffold a repo, write a brief, open Claude Code
  --dry-run               print what would happen, create nothing
```

`mj do` reads, greps and lists freely; **every write, edit and command is shown and
confirmed first**, and paths are confined to the directory you launched from. Credential
files — `.env`, `.ssh/`, `*.pem` and friends — are refused outright, including reads,
because a read means the contents reach a model provider.

### Knowing things

```
mj brief                  fetch, fuse, print and speak what needs you
  --no-speak              print only
  --explain               show which model handled each source, and why
mj activity               what you have been doing on GitHub
  --refresh               fetch new events first
  --days N                how far back to look
mj remember <fact>        write a memory; omit the fact to list them
  --update NAME           replace an existing one
  --forget NAME           delete one
mj sessions               list live / idle / blocked coding sessions
mj resume <id>            jump back into a session, on the right surface
```

### Wiring it in

```
mj config                 what is configured, and which model does what
  --init                  write an annotated config file to edit
mj mic                    measure your microphone, to tune voice input
mj install-hooks          make Claude Code report session state
mj install-trigger        brief automatically on wake / boot / login
mj tray                   the resident tray icon
```

Each has an `uninstall-` counterpart.

## How it is put together

Two different shapes, and the difference is the most interesting thing here.

**The agent is a loop.** It picks the next action from a tool set and keeps going until it
is done or hits a turn cap. This is the only place a model decides what happens next — and
therefore the only place with a confirmation gate. Reads run freely, writes and commands
are shown and confirmed, paths are confined, credentials are refused outright.

**The briefing is a fixed pipeline.** Workers fetch from each source, a router decides
whether a payload needs reducing first, and a fuser writes the single summary. The code
chooses every step; models only produce text. If a source fails it degrades to a plain
list rather than failing.

**A conversation is a model, and the terminal is one consumer of it.**
[`session.py`](majordomo/session.py) owns turns, compaction and the file on disk;
[`chat.py`](majordomo/chat.py) owns the loop, the prompt and the keys. They were one
module, and every bug at the seam was the two disagreeing about the same conversation —
what was stored differing from what you were shown.

## Development

```
pytest
```

976 tests. The suite covers the safety properties directly — path confinement, the
confirmation gate, what compaction keeps — because those are the parts where being wrong
is expensive rather than merely annoying.

## License

MIT
