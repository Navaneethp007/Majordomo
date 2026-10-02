# Majordomo

> A majordomo is the head of a household staff who manages everyone and briefs the master.

A personal assistant that runs on your machine, knows what you have been building, and
does something about it. You talk to it, it answers with your context already loaded, and
it can act — read code, write files, run tests — asking before anything changes.

It is **not** a notification relay, and it is **not** a daemon. Your apps already ping you
"PR merged" / "new ticket". Majordomo does the opposite: it digests those streams and
surfaces only what needs a decision, when you ask it to.

## Install, and the only command you need

```
uv tool install majordomo-cli
mj
```

That is the whole thing. The first `mj` configures itself — a model to talk to, and
optionally the Claude Code hooks and a spoken briefing on wake — and then drops you
straight into the conversation. Every `mj` after that just opens it, with a line telling
you what it already knows and where you left off.

```
$ mj
Majordomo. /help for commands, `mj help` for the CLI, /exit to leave.
(6 memories indexed, 2 loaded in full, activity through 2026-09-26)
Last time: 20260926-1431   14 turns  why is the fuser promoting context items

you › ▏
```

**Why one command.** A tool has one job and a front door made of flags. An assistant is a
place you go, and its front door is itself. Sixteen subcommands meant knowing what you
wanted before you arrived — and meant sixteen names that could never change. One door is
one promise.

**Three names, and they differ on purpose.** You install `majordomo-cli`, you type `mj`,
and you import `majordomo`. `majordomo` was taken on PyPI by an unrelated package, and
plain `mj` is refused by it — untaken but not allowed, which the index will only tell you
at upload time. `[project.scripts]` is independent of all that, so the command is one
short word regardless.

`uv tool` (or `pipx`) rather than plain `pip`, because an app installed into whichever
virtualenv happened to be active disappears when you deactivate it. Plain `pip` still
works and is worth using if you want to import from this — `agent.run`, `brief.run`,
`context.build` are real library surface:

```
pip install majordomo-cli
from majordomo import agent, brief, context
```

Python 3.10+. Windows first; nothing is deliberately platform-locked but nothing else is
tested.

### If you would rather not paste an API key

Setup offers a local model instead. If Ollama is running, everything can point at it: no
key, no network, nothing leaving the machine. Slower, and a small model struggles with the
briefing's rules, but it works end to end. See `api_key_env: ""` in the example config.

### Configuration is optional

Every setting has a default, so there is nothing you must write:

```
mj config           what is in force right now, and which model does what
mj config --init    write an annotated ~/.majordomo/config.yml to edit
mj setup            go through first-run setup again
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
  api_key_env: ""                 # empty means "needs no key" — see below
  chat_model: qwen2.5:14b
  agent_model: qwen2.5-coder:14b
  fallback_model: ""              # nothing to fall back to
```

An **empty** `api_key_env` sends no `Authorization` header at all, which is the only
correct spelling for a local server. Naming a variable is a promise that it holds
something, so a name that is unset is still an error — right for a hosted provider, and
exactly wrong for one on your own machine.

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

**The shipped defaults will go stale.** They are OpenRouter *free* model ids, and free ids
appear and vanish. When one does, the failure says which role was pointing at it and which
line to edit rather than leaving you with a 404:

```
No endpoints found for dots-studio/dots-3-note-preview:free
  … was rejected and will keep being rejected. Free model ids come and go, so
  this is usually a shipped default that has aged out rather than anything you did.
  Set by:
    brain.worker_model   compress each source
    brain.chat_model     mj ask, mj chat
  Edit ~/.majordomo/config.yml — `mj config` lists all six roles.
```

Setup does not ask you to pick models. Choosing between model ids is not a question
anyone can answer in their first minute, and it is the same "know what you want before you
arrive" problem the single front door removes.

[majordomo/config.example.yml](majordomo/config.example.yml) explains every setting.

Optional extras, none required for text:

| Extra | For |
|---|---|
| `majordomo-cli[nvidia]` | speech, in and out, via NVIDIA Riva (gRPC, hence separate) |
| `majordomo-cli[voice]` | microphone capture — the one dependency speech *input* costs |
| `majordomo-cli[documents]` | text out of PDFs and Word files |
| `majordomo-cli[tray]` | the resident tray icon |

Adding one later means reinstalling with it named, e.g.
`uv tool install --force "majordomo-cli[voice]"`. Every "not installed" message tells you the line.

## The door

```
mj                        the conversation. This is the product.
mj help                   list these commands
```

Piped input is answered once and exits, so the door is scriptable too:

```
$ echo "what did I ship this week?" | mj
```

Inside the session:

| | |
|---|---|
| `/context` | what memory and activity are loaded, and how large it has grown |
| `/clear` | start fresh, keeping the loaded context |
| `/read PATH` | hand it a document — text, PDF or Word — to talk about |
| `/remember` | propose what from this conversation is worth keeping |
| `/agent TASK` | put the agent to work without leaving the conversation |
| `/build NAME` | scaffold a repo from this conversation |
| `/voice` | speak your next message instead of typing it |
| `/help` | the list |

These are cheap to rename precisely because nobody scripts against them — which is the
other half of why there is one door.

`/read` is the one that reaches the disk from a plain conversation. Chat has no tools by
design, and this is not a hole in that: the agent is confined to a project directory
because a *model* chooses the paths, and here you typed one. What does still apply is the
credential denylist — `.env`, `.ssh/`, `*.pem` are refused, because a read means the
contents reach a model provider — along with the same 8,000-character cap the agent's
reads get, so a long document cannot quietly swallow the conversation.

The document's text is also **fenced**, and this is the part worth knowing. The agent's
file reads come back as `role: "tool"`, a channel the model knows is machine output;
`/read` has no tool call to attach to, so its text can only arrive as a user turn — the
highest-trust channel there is. But "hand it a document" usually means a document somebody
*sent* you, so its author is generally not you. The text is therefore wrapped in
`<<<DOCUMENT name>>>` … `<<<END DOCUMENT>>>`, the system prompt says that region is
material to discuss and never instructions to follow, and three markers are defanged
inside it: the closing fence (a fence a document can close is not a fence), `NEEDS_AGENT:`
(or a document picks the task you get asked to approve) and `REMEMBER:` (or it writes
itself a memory that replays forever).

## Commands

The stable set. Scripts and the scheduled trigger depend on these, so they are the ones
that will not move:

```
mj brief                  fetch, fuse, print and speak what needs you
mj ask <question>         one question, answered with your context loaded
mj do <task>              the agent works in the current directory
mj chat                   the session, with --resume and --list
mj config                 what is configured, and which model does what
mj setup                  first-run setup, again
```

Everything else — `sessions`, `resume`, `review`, `start`, `mic`, `activity`, `remember`,
`install-hooks`, `install-trigger`, `tray` — still works exactly as before and is
documented below, but is deliberately absent from `mj --help`. Treat it as internal: it
may be renamed or moved inside the session without notice.

### Doing things

```
mj do <task>              the built-in agent works in the current directory
  --directory, -C DIR     work somewhere else
  --yes                   approve every write and command without asking
mj review <path>          open Claude Code on a repo with /code-review
mj start <idea>           scaffold a repo, write a brief, open Claude Code
  --dry-run               print what would happen, create nothing
```

The agent has eight tools: read, list, grep, write, edit, run a command, **git**, and
**GitHub** via the `gh` CLI. So `mj do "commit this"`, `mj do "open a PR for this branch"`
and `mj do "what do the comments on PR 12 say?"` all work.

What it may do without asking is the interesting part, because it is not simply "reads":

| | |
|---|---|
| runs freely | read, list, grep · `git status`, `log`, `diff --stat`, branch listings · `gh pr view`, `pr list`, `issue view`, and `gh api` for GET of a *named path* |
| shown and confirmed | write, edit, any shell command · `git commit`, `add`, `checkout`, `push` · **and git commands that print file contents** — `show`, bare `diff`, anything with `-p` |
| refused outright | credential files, however they are reached · force-pushing · `gh auth`, `gh secret` |

Two of those rows are worth the explanation.

**Content-printing git commands are gated even though they only read.** `read_file`
refuses `.env` outright — a read means the contents reach a model provider, and a leaked
key cannot be un-leaked — but `git show HEAD:.env` prints the same bytes. Treating git
reads as free would have opened a new ungated path to exactly what the denylist exists to
stop. So the denylist reaches into git's arguments too, and `-p` asks first.

**Force-pushing is refused, but `--force-with-lease` is allowed.** The lease form fails
rather than clobbering when the remote has moved, so refusing it would only push the agent
toward the dangerous flag.

Paths stay confined to the directory you launched from, and the git tool additionally
refuses a repository that sits *above* it, since git otherwise walks upward out of the
project. Both git tools take an argument list rather than a command line, so there is no
shell: no `&&`, no redirection, no substitution.

`--yes` approves every local write without asking, and still refuses to post to GitHub
when stdin is not a terminal. A bad commit is recoverable with git; a comment under your
name on somebody else's pull request is not.

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

Setup offers the two that touch your machine. These are the manual equivalents:

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

1357 tests. The suite covers the safety properties directly — path confinement, the
confirmation gate, what compaction keeps — because those are the parts where being wrong
is expensive rather than merely annoying.

## License

MIT
