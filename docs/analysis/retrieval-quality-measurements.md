# Why our coverage is duller than a curated newsletter — the measurements

Measured 2026-09-09 against live PROD (2320 cached articles, 26
subscribers). Written because a subscriber asked a specific question —
*"TLDR sends me things worth clicking; why don't you?"* — and handed over
five example links. Chasing those five links produced findings that
falsified two of this session's own hypotheses, so the numbers are
recorded here rather than only the conclusions.

**TLDR is deliberately NOT ingested as a source.** It is the held-out
baseline. Ingesting it would cap our coverage at theirs and destroy the
only independent yardstick we have for measuring the gap.

## The five links, and whether we had them

| Link | In cache? |
|---|---|
| gizmodo.com — fruit fly brain plays Doom | ✗ (4 other Gizmodo pieces, via `gnews`) |
| oruk.ai — taught a fruit fly to read emotion | ✗ (0 from this domain) |
| ifa-berlin.com — humanoid robots at IFA | ✗ (the 2 "hits" are CNET/ZDNet writing *about* IFA) |
| foxnews.com — autonomous excavators, empty cabs | ✗ (5 other Fox pieces, all Health/Crypto) |
| blog.sshh.io — I asked 100 agents to hack me | ✗ (0 from this domain) |

Keyword confirmation across the whole cache: `excavator` 0, `Lightpanda`
0, `hack me` 0.

Gizmodo and Fox News appear at all only because `gnews`/`newsapi` happened
to surface unrelated pieces from them. **Neither is a configured source.**

## Finding 1 — a fifth of the corpus is off-thesis

Category distribution, aggregator APIs vs. our own curated feeds:

| | Aggregators (`gnews`/`newsapi`/`perigon`), n=471 | Curated RSS excl. arXiv, n=1577 |
|---|---|---|
| 1st | Science 89 | **AI 637** |
| 2nd | Technology 87 | Technology 512 |
| 3rd | Consumer 82 | Business 319 |
| 4th | Research 78 | Software 281 |
| 5th | **Government 78** | Research 247 |
| 6th | Business 66 | Consumer 228 |
| 7th | **Gaming 57** | Science 177 |
| 8th | **Politics 55** | Finance 159 |
| … | **AI ranks 11th (37)** | |

The aggregators are **20% of the corpus** and their top category is
general Science, with Government, Gaming, Politics and Entertainment all
above AI. Concrete examples pulled on 2026-09-04..09: *"Your daily coffee
habit could be taking a toll on your bones"*, *"New tick-borne virus
discovered in China"*, *"The 'American Cheetah' Wasn't a Cheetah"*.

**This is not a filter failure — we ordered it.** The sections we request
are `gnews: technology, business, science, world` and
`newsapi: technology, business, science, health`.

## Finding 2 — the registry is weighted 6:1 against community discovery

`news_sources.py`'s `SOURCE_SECTIONS`:

```python
"arxiv":      ["cs.AI", "cs.LG", "cs.RO", "cs.CR", "quant-ph", "physics.optics"],  # 6
"hackernews": ["front_page"],                                                      # 1
```

The corpus mirrors it almost exactly: **arXiv 272, Hacker News 72**
(3.8:1). HN is our only channel to personal blogs, project launches and
indie research — the layer three of the five links came from.

Ingestion is healthy, not broken: `hackernews:front_page`'s newest article
was same-day at time of measurement.

**Lightpanda was on HN at 319 points** ("Show HN: Lightpanda, an
open-source headless browser in Zig"), plus follow-ups at 212 and 199. We
had the channel and did not pull the section it lives in.

## Finding 3 — only one source carries a popularity signal, and we discard it

| Source | Popularity signal |
|---|---|
| **Hacker News** (Algolia) | ✅ `points`, `num_comments` — a real human vote |
| NewsAPI | ⚠️ `sortBy=popularity` sorts, but no numeric field is returned |
| GNews / arXiv / all RSS | ✗ none exists |
| Perigon | ❓ our adapter never touches its scoring fields; unverified |
| Lobsters (candidate) | ⚠️ RSS has none; the JSON API exposes `score` |

`news_adapters/hackernews.py` maps `title`/`link`/`source`/`summary`/
`published` only — **`points` is dropped on the floor.** Its `summary` is
also always `None`, so HN articles are embedded from their title alone.

## Finding 4 — the interest *definition* is a far bigger lever than expected

Retrieval is `query_text = definition or topic`, where `definition` is an
LLM-generated expansion cached in `interest_query_expansions`. Same corpus
(2503 embedded articles), same interest, three query texts — rank of three
target articles:

| Article | bare `"AI"` | auto-generated `AI Agent` def | hand-written demo-flavoured def |
|---|---|---|---|
| RoboCousin: Build Your Own Simulation Playground | 467 | 23 | **2** |
| A new game demonstrates quantum advantage | 1009 | 786 | **12** |
| Second complete map of a fruit fly brain | 874 | 1303 | 1277 |

**Rewriting one paragraph moved an article from rank 467 to rank 2.** The
definition, not the interest word, decides what a subscriber receives —
and it is generated behind their back and never shown to them.

Two specific defects fall out of this:

- The auto-generated definitions encode a **vendor/enterprise prior**. The
  live `AI Agent` definition reads *"...tool calling, LangChain, AutoGen,
  CrewAI, ReAct prompting, RAG... deployed as virtual assistants or
  workflow automation tools"* — so it retrieves *"How agents are
  transforming work"*, not *"I asked 100 agents to hack me."*
- **A subscriber's `AI` interest has no definition at all** (`(none)` in
  the table), so `definition or topic` falls back to embedding the
  two-character string `"AI"` — a near-meaningless retrieval vector. It
  returns *"Introducing AI-as-a-Service"*, *"AI for Food Allergies"*.

## Finding 5 — but genre cannot be expressed as a definition

This one falsified the session's own follow-up hypothesis. Cosine of each
headline against the demo-flavoured definition:

```
+0.234  We taught a fruit fly to read human emotion          ← wanted
+0.167  I asked 100 agents to hack me                        ← wanted
+0.151  Autonomous excavators digging with empty cabs        ← wanted
+0.146  [baseline] Nvidia's $12.9B Hugging Face deal         ← NOT wanted
+0.136  [baseline] Trump administration sides with OpenAI    ← NOT wanted
+0.130  Google mapped a fruit fly's brain, now playing Doom  ← THE flagship link, last
```

The subscriber's single best example scores **below a corporate
acquisition story**, despite its headline containing "playing Doom" and
"Super Mario 64". Total spread between wanted and unwanted is ~0.09 — not
enough to rank on.

Meanwhile the *topical* `AI Agent` definition scored *"I asked 100 agents
to hack me"* at **+0.342**, the highest value anywhere in the table.

**Conclusion: the definition lever works through topical vocabulary
overlap, and it works best when the definition is topically specific.**
Writing a genre ("hands-on experiments, not funding news") into a
definition is a weak discriminator, because static embeddings encode
subject matter, not story shape.

A proposal to add an explicit `story_kind` classifier label was
**withdrawn** on this evidence: the argument for it rested on the
ars_technica fruit-fly article, which is not the article the subscriber
actually asked for. Revisit only if a measured residual gap survives the
definition work.

## Finding 6 — the taxonomy has no duplicate gate

245 active categories, containing at minimum:

| Class | Examples |
|---|---|
| Case variants | `Real Estate` / `Real estate` / `RealEstate`, `Social media` / `Social Media`, `Private equity` / `Private Equity` |
| Plural variants | `Semiconductor(s)`, `Drone(s)`, `Disaster(s)`, `Consumer(s)`, `Pharmaceutical(s)` |
| Spelling errors | `Techology`, `Techonlogy`, `Minning`, `Entrepeneurship` |
| Synonyms | `Math` / `Mathematics`, `Labour` / `Labor`, `Ecommerce` / `E-commerce`, `Telecom` / `Telecommunications` |

`category_ops.normalize_category_name` does whitespace/colon cleanup and
truncation only — and says so: its docstring scopes it to *"makes a
model-proposed label safe to round-trip through a Telegram callback."*
There is no case folding, no stemming, no spell check, and no comparison
against existing names.

The only gate is a human admin approving at 5 sightings. Spotting that
`Semiconductors` duplicates `Semiconductor` among 245 existing names is
precisely what humans are bad at.

The table already has a `centroid` column and a `merged_into` column —
the machinery for similarity comparison and for merging both exist, and
neither is consulted at proposal time. (`centroid` was reserved in
`taxonomy-and-admin-plan.md` A6 for a different purpose: nearest-centroid
*classification*.)

## What each finding became

| Finding | Where it goes |
|---|---|
| 4, 5 | `docs/plans/interest-definition-plan.md` — make the definition visible and refinable |
| 6 | `taxonomy-and-admin-plan.md` §A8 — a duplicate gate + one-off merge |
| 1, 2, 3 | `local-news-cache-plan.md` — configurable sections, pull order, retain HN points |
