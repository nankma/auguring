# Autonomous Technology-Trend Intelligence Agent

**An LLM agent that monitors 10 technology news sources, works out what's
genuinely new, and delivers a personalized trend briefing on Telegram — on
a schedule, without being asked.**

Built on LangChain with DeepSeek, running in production on Oracle Cloud.
Small in scope by design; complete in lifecycle — architecture, security,
deployment, observability, testing, and live incident response are all
built and in use.

<p align="center">
  <img src="docs/images/digest-briefing.jpg" alt="A scheduled Telegram briefing grouped by the subscriber's topics — AI and Robotics — each item synthesized across several sources with links" width="45%">
  <img src="docs/images/digest-sources.jpg" alt="The end of a briefing, where the agent notes its sources on a topic are weeks old and recommends checking a live feed" width="45%">
</p>

Each item is **synthesized across sources** rather than listed — the first
entry above merges New Scientist, Wired, and TechCrunch coverage into one
paragraph, with every claim carrying its links.

> **Why "Auguring"** — an augur read scattered signs — the flight of
> birds, the weather, whatever was in front of them — and turned them
> into a reading of what was coming. That's closer to what this service
> actually does than "watching" is: the value isn't in seeing every
> source, it's in synthesizing what they add up to. Fitting for
> something that reads across more sources than a person reasonably
> can, and tells you what they mean rather than just listing them.

---

## Why it exists

Keeping current on a few areas of technology meant working through a dozen
sites and forums regularly — Hacker News, arXiv, company engineering
blogs, the tech press. Most of it was either duplicated across all of them
or irrelevant. The reading wasn't the expensive part; the **filtering**
was.

So: something that makes that pass for you, works out what's actually new,
summarizes the *trends* rather than the headlines, and delivers the result
instead of waiting to be asked.

## Features

| | |
|---|---|
| **Trend reports on demand** | Ask about a company, product, or topic; get a synthesized report citing real sources |
| **Guided interest discovery** | Don't know what to follow yet? A conversation shows you real recent headlines and narrows down from there, instead of asking you to already know a topic name |
| **Personalized interests** | Multi-user — each subscriber sets their own topics, which steer every query |
| **Reply language** | Set once, applies to everything after, including script variants (Traditional vs. Simplified Chinese) |
| **Scheduled push digests** | Per-user interval, deduplicated so the same article is never sent twice |
| **Access control** | Admin-approval workflow via a separate bot; not an open service |

Everything is controllable **two ways** — a slash command or plain natural
language ("add robotics to my interests", "start pushing me news every 6
hours"). Natural language isn't decoration: voice input is a planned
direction, and voice has no slash commands.

### Commands

| Command | Does |
|---|---|
| `/start` | Register — first-time users enter the approval queue |
| `/interests` | Show, set, or clear your topics |
| `/language` | Show, set, or clear your preferred reply language |

Anything else is treated as a request and routed by the agent — including
open-ended ones like *"help me figure out what to follow"*, which starts
the guided-discovery conversation above rather than requiring you to name
a topic up front.

## How it works

Every message runs through the same three stages:

```
message → [regex pre-filter] → Router (LLM) ─┬→ Route A: Research  ─┐
                                             │   search + synthesize │
                                             └→ Route B: Settings   ─┤
                                                 update subscriber   │
                                                                     ↓
                                                          Verify (LLM) → send
```

The **router** decides both whether the message is in scope and which
route it takes — one call doing a safety gate and intent routing together.
**Verify** independently checks what the model actually wrote before it
reaches the user, because some failure modes are only visible in the
output.

The scheduled digest deliberately **does not** let the agent choose what
to fetch. Selection and deduplication are ordinary deterministic code; the
LLM is used only to write prose from a fixed article list. That makes
repeats impossible by construction rather than unlikely by persuasion.

📖 **[Full technical write-up →](docs/system-overview.md)** — architecture,
design decisions, measurements, and the problems hit along the way.

## Quick start

Requires a [Miniforge](https://github.com/conda-forge/miniforge) install.
Dependencies are conda-forge packages declared in `environment.yml` —
there is no `requirements.txt` path.

```powershell
conda env create -f environment.yml     # first time only
conda activate myfirstagent
```

**CLI (no Telegram needed):**

```powershell
$env:DEEPSEEK_API_KEY = "<your-deepseek-key>"
python agent.py
```

**Telegram bots** — needs two bots from [@BotFather](https://t.me/BotFather),
one public and one for admin approvals:

```powershell
$env:DEEPSEEK_API_KEY = "<your-deepseek-key>"
$env:TELEGRAM_BOT_TOKEN = "<public-bot-token>"
$env:ADMIN_BOT_TOKEN = "<admin-bot-token>"
$env:ADMIN_CHAT_ID = "<your-numeric-telegram-user-id>"
python combined_bot.py
```

`combined_bot.py` runs both bots and the push scheduler in one process —
see [why](docs/system-overview.md#appendix-b--difficulties-and-how-they-were-solved).
`bot.py` and `admin_bot.py` still run standalone for local development.

> **Adding a dependency?** Use `mamba`, not `conda` — conda's classic
> solver has repeatedly stalled on this dependency tree. Then add it to
> `environment.yml`.

## Testing

```powershell
pytest
```

**160 tests, ~2.5 seconds, no API cost.** The model is dependency-injected
— production passes a real client, tests pass a scripted fake — so the
suite needs no network, no API key, and has no flakiness from a live
model.

What unit tests structurally *cannot* cover is real model behavior. That's
handled by a 13-case checklist run against the live service after every
deployment; see the `build-locally-deploy-remotely` skill.

## Deployment

One VM on Oracle Cloud's Always Free tier, running the bot container.
LLM tracing goes to [Pydantic Logfire](https://pydantic.dev/logfire), a
cloud-hosted OpenTelemetry backend, so no second VM is needed for it.

```bash
docker build -t myfirstagent-bot .
docker save myfirstagent-bot:latest | ssh -i <key> ubuntu@<vm-ip> "sudo docker load"
```

**Always build locally and transfer** — never build on the VM, which is a
1 GB free-tier shape. Secrets are fetched from OCI Vault at container
startup using Instance Principal authentication, so no credential is baked
into the image or passed as a plaintext environment variable.

Full setup, networking, and secrets configuration: **[deployment-plan.md](docs/plans/deployment-plan.md)**.

## Project structure

| File | Role |
|---|---|
| `agent.py` | Agent, tools, layered prompt composition, CLI |
| `bot.py` | Telegram handling, message formatting, push scheduling |
| `admin_bot.py` | Separate bot carrying the approve/deny controls |
| `combined_bot.py` | Runs both bots in one process — the deployed entry point |
| `guardrails.py` | Router and output checks |
| `news_sources.py` | Pluggable registry of the 10 news sources |
| `news_push.py` | Scheduled digest: fetch, deduplicate, write, send |
| `users_db.py` | SQLite store — approvals, interests, language, push settings |
| `telemetry_monitor.py` | Alerts the admin if tracing goes unreachable |
| `tools/` | Build and authoring utilities — see [tools/README.md](tools/README.md) |

## Documentation

**Start here:** [System overview](docs/system-overview.md) — the full
technical write-up.

| Doc | Covers |
|---|---|
| [system-overview](docs/system-overview.md) | Architecture, system design, quality assurance, difficulties solved |
| [observability-and-debugging](docs/reference/observability-and-debugging.md) | Diagnosing the live service; querying traces |
| [security-plan](docs/plans/security-plan.md) | Security review — findings, what's fixed, what's outstanding |
| [deployment-plan](docs/plans/deployment-plan.md) | Cloud setup, CD design, deployment workflow |
| [telemetry-and-testing-plan](docs/plans/telemetry-and-testing-plan.md) | Test strategy and telemetry design |
| [guardrails-plan](docs/plans/guardrails-plan.md) | Four-layer safety design and reliability measurements |
| [context-management-plan](docs/plans/context-management-plan.md) | Layered prompt design and the router |
| [bot-features-plan](docs/plans/bot-features-plan.md) | Feature designs — built and planned |
| [ai-news-sources](docs/current/ai-news-sources.md) | Source registry; how to add one |

**Deferred by decision, with reasoning recorded:**

| Doc | Decision |
|---|---|
| [multi-channel-plan](docs/plans/multi-channel-plan.md) | LINE support — on hold; its free tier caps push at 200/month account-wide |
| [data-layer-plan](docs/plans/data-layer-plan.md) | Moving off SQLite — deferred until a second host actually needs the data |
| [model-portability-plan](docs/plans/model-portability-plan.md) | Dynamic model switching — no gateway service needed; the injection seam already exists |

## Changelog

User-visible changes, newest first. Everything else — refactors,
telemetry, deploy tooling — is tracked in commit history and `docs/plans/`
instead of here.

- **2026-09-08** — **Find my interests**: a new guided conversation for
  subscribers who don't already know what to follow. Shows real recent
  headlines and narrows down from there — say *"help me figure out what
  to follow"* to start it. ([#90](https://github.com/nankma/auguring/pull/90),
  fix: [#91](https://github.com/nankma/auguring/pull/91))
- **2026-09-05** — Search got much faster: on-demand queries now read
  from a locally indexed vector cache instead of scanning every article,
  cutting typical response time from tens of seconds to under one.
  ([#85](https://github.com/nankma/auguring/pull/85),
  [#86](https://github.com/nankma/auguring/pull/86))
- **2026-08-25** — Fixed a bug where adding several interests in one
  message ("add AI agent, AI coding, and robotics") could silently drop
  or merge some of them.

## Status

A working pilot, live and serving real users — not a production service at
scale. Known limitations are documented honestly rather than omitted: see
[Appendix B.2](docs/system-overview.md#b2-known-limitations) for what it
can't currently do, each with an explicit trigger for when it must be
fixed.

## License

[MIT](LICENSE)
