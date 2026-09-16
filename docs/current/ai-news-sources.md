# News Sources

`news_sources.py` is a pluggable source registry — free/no-key sources
are always enabled; key-gated sources turn on automatically once their
env var is set. It powers `news_ingest.py`'s scheduled background pulls,
which populate the local cache both `news_push.py`'s digests and
`agent.py`'s `search_news` read (`agent.py`'s `search_news` stopped
calling any source here live, 2026-09-04, and stopped being a
LangChain tool at all, 2026-09-05 — see
`docs/plans/local-news-cache-plan.md` item 5). This doc tracks what's
wired up, what's just documented for later, and how to add more.

Originally AI-industry-only. Broadened 2026-08-13 to general tech/business/
finance press, after a real gap: a subscriber asked about a specific
company (AAOI, a fiber-optic components maker) and no source in the
registry covered anything outside AI-industry blogs and community boards.

## Source classes

Every entry in `SOURCE_REGISTRY` is tagged with a class. `enabled_sources()`
itself still doesn't branch on it, but `news_ingest.py` now does (see
"Since-based ingestion" below) — it matters for two reasons worth knowing
before relying on a source's result count: **most sources here don't
actually filter by the search query, and most can't be asked "give me
everything since X" either.**

| Class | Meaning | Filters by query? |
|---|---|---|
| **forum** | Community-curated discussion board (submissions + votes), not edited articles | Yes — real search |
| **api** | JSON REST API with real query-based search | Yes — real search |
| **rss** | Standard RSS/Atom feed | **No** — returns the feed's latest N items regardless of what was asked |

Of the 21 currently-enabled sources, only **5** (`hackernews`, `arxiv`, plus
the three key-gated `api`-class sources when a key is set) do real
filtering. The other 16+ are `rss`-class and always return their latest
items whether or not any of them are actually relevant — this is why a
big pull count from `news_ingest.py` is not proof any of it is on-topic;
`news_push.select_candidate_articles`'s category/relevance filters and
`news_embed.filter_by_relevance` (also used by `search_news`, see
`docs/plans/local-news-cache-plan.md` item 5) are what actually narrow
the cache down downstream of ingestion.

## Since-based ingestion (added 2026-08-16)

`news_ingest.py` used to cap every source at a flat top-5 per query
(`MAX_RESULTS_PER_SOURCE`), regardless of how much was actually new since
its last pull — a real bottleneck on an active source: anything past the
first 5 was silently discarded that cycle, gone until (if ever) it
resurfaced in a later "latest 5". Now the 5 `forum`/`api`-class sources
fetch "everything since this source's last successful pull" instead,
verified live 2026-08-16 against each provider's real API (not assumed
from docs):

| Source | Server-side date filter | Verified live | Notes |
|---|---|---|---|
| **Hacker News** | `numericFilters=created_at_i>X` (Algolia) | ✅ 45 hits in a 6h window, all strictly after the cutoff | |
| **arXiv** | `submittedDate:[X TO 99991231235959]` range in `search_query` | ✅ syntax confirmed working | Real caveat: arXiv's own indexing lags multiple days (an unfiltered query on 2026-08-16 returned nothing newer than 2026-08-13) — a short since-window often legitimately returns nothing. Not a bug, and no worse than the old flat cap on a source this slow. |
| **GNews** | `from=` (ISO 8601) | ✅ 30 articles in a 24h window | `max` is still capped at 10/request by GNews's own free tier regardless of what's asked. |
| **NewsAPI** | *(deliberately not used)* | ❌ found broken for this use case | `from=` **works syntactically** but the free "Developer" tier has an undocumented ~24-36h article delay — `from=<24h ago>` returned 0 results live, `from=<36h ago>` returned 380. Since NewsAPI is pulled once every 24h, a server-side since-filter would frequently return nothing. Handled with client-side filtering instead (see below), which doesn't have this failure mode. |
| **Perigon** | *(deliberately not used)* | Not tested — no API key available | Same caveat as its response-shape mapping elsewhere in this doc — unverified, not trusted without a key to check against. |

**Two mechanisms, not one.** A server-side date filter (where verified
above) is applied as an efficiency optimization — smaller payloads,
less wasted budget on rate-limited sources. But the actually-authoritative
filter is a **client-side check in `news_ingest.py`** applied to every
`forum`/`api`-class source's results regardless: drop anything with
`published_dt` at or before the cutoff. This is what makes NewsAPI/Perigon
work correctly despite having no server-side filter at all, and it's also
the backstop if a server-side filter above ever silently misbehaves.
`rss`-class sources are unaffected by any of this — a plain feed has no
query or date-range parameter to ask for "since X" in the first place, so
there's nothing to switch to since-based fetching for them.

**The cutoff is the newest article's own `published_dt` actually seen
from that source, not when the ingestion job last ran.** A design
correction made the same day, after the job-run-time version was found to
have a real failure mode: `last_pulled_at` (wall-clock job time) advances
every cycle regardless of whether anything new was found, so an article a
source indexes with a delay (exactly NewsAPI's ~24-36h delay above) could
have its `published_dt` fall *behind* a since-cutoff that already moved
past it by the time the source finally surfaces it — silently skipped
forever, not just delayed. Fixed by tracking a separate per-source value
(`source_state_ops.get_source_last_article_dt`/`set_source_last_article_dt`) that
only advances to the max `published_dt` actually observed each cycle, so
it can never outrun what's genuinely been seen the way a wall-clock
timestamp can.

**Their own top-N cap was raised instead, same day**: `rss`-class sources
went from a flat 5 to 200 (`news_ingest.MAX_RESULTS_PER_SOURCE_RSS`) — the
5 was arbitrary and, per a real subscriber report, was cutting pushed
digests down to a handful of items even when a feed had more genuinely
new content available. 200 comfortably exceeds what any registered feed
actually carries (most run 20-50 entries per the content-depth
investigation below), so this is effectively "take everything the feed
has" now, not a real limit.

**A new problem that cap raise created, and its fix**: at 200/feed, most
of a cycle's fetch is typically the *same* items as the previous cycle
(feeds don't turn over that fast) — without a dedup check, every one of
them would go through a real, paid DeepSeek classification call every 4
hours for no reason (`news_cache.write_article`'s overwrite-by-link-hash
already makes a redundant *write* harmless, but a redundant
*classification call* isn't free the same way). Fixed by loading every
currently-cached link once per ingestion cycle and skipping
classification/caching for anything already present. Both the
newly-cached and already-cached counts are logged per source and per
cycle specifically so `MAX_RESULTS_PER_SOURCE_RSS` can be tuned again
later from real data (`docker logs`) rather than guessed at a second
time.

## Download lag per source (measured 2026-08-19)

How far behind an article's own publication time we actually download it,
measured across 2,253 cached articles by comparing `fetched_at` against
`published_dt`. This matters more than it looks: until 2026-08-19,
`news_push` filtered candidates on `published_dt`, so **any source with a
real publication delay was structurally excluded from digests** no matter
how good its content (see `news_push.select_candidate_articles`).

| Source | Median download lag | Note |
|---|---|---|
| `hackernews` | 1.7 h | |
| `techradar` | 2.6 h | representative of the RSS sources |
| **`gnews`** | **12.8 h** | matches the documented 12-hour free-tier delay exactly |
| **`arxiv`** | **12.8 h** | arXiv's own indexing lag, not a tier restriction |
| **`newsapi`** | **32.1 h** | matches the 24–36 h free-tier delay measured 2026-08-16 |

The concrete cost of the old rule: **227 GNews articles sat in the cache,
correctly fetched and classified, with zero of them eligible for any
digest.** Measured against the same snapshot after the fix, all 227 are
eligible — because eligibility no longer consults a date at all. A date
now only ranks; `already_pushed_links` alone decides what a subscriber has
seen (see `news_push.select_candidate_articles`).

Delayed sources still rarely *win* the ranking, since it is publication
order and they are 12–32 h behind by construction. The difference is that
they are no longer disqualified: the candidate pool drains in publication
order, so an unsent article keeps its place until it is actually sent or
ages out of the cache. Making delayed-but-valuable content win on merit
rather than recency is the separate ranking question tracked in
`docs/analysis/news-ranking-plan.md`.

## Content depth per source (investigated 2026-08-13)

Prompted by a real question: besides the title, what does each source
actually give us, and where is content genuinely unavailable versus just
being discarded by our own code? Every number below was checked live,
against the real, uncapped field — not assumed from a source's docs.

### Fixed: `_fetch_rss` was discarding the feed's own description

`_fetch_rss` previously hardcoded `summary=None` for every RSS source,
throwing away whatever the feed's `<description>` provided regardless of
content. Fixed via `_clean_summary` (strips embedded HTML, normalizes
whitespace). **Still open: what cap to apply** — see the table below,
where two sources make a fixed 300-char cap actively wrong.

### The full picture, uncapped

| Tier | Source | Raw length | Currently kept | Notes |
|---|---|---|---|---|
| **Substantial — likely near-full article text** | VentureBeat AI | **13,384 chars** | 300 (2%) | Feed embeds most/all of the article body directly (`content:encoded`-style), not a lede. Discarding 98% of it. |
| | Computerworld | **4,642 chars** | 300 (6%) | Same pattern. |
| **Full source-native content, source doesn't truncate — we do** | arXiv | ~1,700 chars (varies per paper) | 300 (~18% in the checked example) | The *complete* abstract, at zero extra cost — same API call already being made. The discarded portion is typically where the paper's actual method name and results live, not the problem statement. |
| **Real editorial lede, moderately truncated** | Guardian Business | 750 | 300 (40%) | |
| | Guardian Technology | 550 | 300 (55%) | |
| | MIT Technology Review | 351 | 300 (85%) | Cap barely bites. |
| **Short dek/summary, genuinely short by design — cap rarely or never bites** | OpenAI Blog (157) · BBC Business (108) · BBC Technology (105) · MarketWatch (115) · Economist Business (61) · Economist Sci&Tech (74) · Wired Business (141) · The Register (82) · ZDNet (120) · TechRadar (117) · TechCrunch AI (54) · Engadget (58) | — | — | These 12 sources are already giving us everything they have; the cap is not the constraint here. |
| **Content only sometimes** | Hacker News | Full, untruncated `story_text` for ~5% of results (Ask HN/Show HN self-posts, checked: 1/20 in a live sample) | Not mapped at all today | The other ~95% are external link posts — HN's own API has nothing beyond title+url for those; it doesn't host the linked content. |
| **Title only — genuinely nothing else in the feed** | Hugging Face Blog | — | — | Confirmed via `feedparser`: no `summary`/`description`/`content` field exists in this feed at all. |
| | Nikkei Asia | — | — | Same — the RDF feed provides title/link/date only. |

### Not yet enabled — documented behavior, not live-verified (no key)

| Source | What's mapped now | What's actually available |
|---|---|---|
| NewsAPI | `description` only | Also has a `content` field, but their free/Developer tier truncates it to ~200 chars with a `"… [+N chars]"` marker. Full content needs a paid plan. |
| GNews | `description` only | Same pattern — `content` exists, free tier truncates it similarly. |
| Perigon | `summary` only | Least certain of the three — no confident documentation on free-tier content completeness; would need a trial key to check rather than assume. |

### What this means for the 300-char cap

The cap was arbitrary — nothing in the code chose it deliberately, and it
now demonstrably cuts VentureBeat/Computerworld/arXiv well before the
content that matters (arXiv's actual result, in the one case checked in
detail, lands in the discarded 82%). **Not yet fixed** — raising or
dropping the cap is a small, low-risk follow-on (same shape as the
summary-discard fix above: expose data already being fetched, at zero
extra network cost), tracked as a pending item, separate from the API-key
work below since it doesn't depend on it.

For the three structurally content-less/near-content-less sources
(Hugging Face Blog, Nikkei Asia, and the 95% of Hacker News that's link
posts) — and for anything a lede genuinely doesn't mention, like a
consequence reported deeper in an article body — closing that gap needs
either full-page scraping or a paid API tier. See
`docs/plans/local-news-cache-plan.md`'s open question on this; it's a
materially bigger decision (scraping, paywalls, bot defenses, legal
posture, recurring cost) than anything on this page.

## Enabled now (free, no key required)

### AI-industry press (original scope)

| Source | Class | Endpoint | Notes |
|---|---|---|---|
| **Hacker News** | forum | `https://hn.algolia.com/api/v1/search_by_date` (Algolia HN Search) | Use `search_by_date`, not the default `/search` — the latter ranks by relevance/points and surfaces old high-upvote posts instead of recent ones. Community-submitted, so quality/relevance is mixed (raw signal, not editorial). |
| **arXiv (cs.AI)** | api | `http://export.arxiv.org/api/query` | `search_query=cat:cs.AI`, `sortBy=submittedDate`, `sortOrder=descending`. Atom XML response, parsed with `feedparser`. Found 2026-09-14/15: 429s/timeouts on most of its 6 sequential per-cycle section calls — root cause was our own `REQUEST_DELAY_SECONDS` (1.1s) being under arXiv's own documented Terms of Use ("no more than one request every three seconds" — info.arxiv.org/help/api/tou.html), not a code bug or an arXiv-side problem. Fixed by raising the shared delay to 3.0s; sections stay separate (see "Consequences worth knowing" below for why combining them into one OR'd query was considered and rejected). |
| **OpenAI Blog** | rss | `https://openai.com/news/rss.xml` | Standard RSS 2.0. |
| **Hugging Face Blog** | rss | `https://huggingface.co/blog/feed.xml` | Standard RSS 2.0. |
| **TechCrunch AI** | rss | `https://techcrunch.com/category/artificial-intelligence/feed/` | Standard RSS 2.0. |
| **VentureBeat AI** | rss | `https://venturebeat.com/category/ai/feed/` | Standard RSS 2.0. **Broken 2026-09-14/15 on, not a frequency problem**: the whole `venturebeat.com` domain (now on Vercel) returns a site-wide bot-challenge `429` to every automated request — confirmed live from two unrelated IPs, with both a custom and a real-Chrome User-Agent, on `/`, `/sitemap.xml`, and the feed itself alike. VentureBeat has no public API to apply for, and their old FeedBurner-based `feeds.venturebeat.com` alternative is dead (DNS still points at `feeds.feedburner.com`, which 404s — Google decommissioned that service). Throttled to `interval_hours: 24` (was the 4h default) purely to monitor for when their WAF config gets fixed, not to work around a real rate limit. |
| **MIT Technology Review** | rss | `https://www.technologyreview.com/feed/` | Main feed, not AI-filtered — but heavily AI-weighted anyway. |

### Mainstream press — Business/Finance (added 2026-08-13)

| Source | Class | Endpoint | Notes |
|---|---|---|---|
| **BBC Business** | rss | `http://feeds.bbci.co.uk/news/business/rss.xml` | |
| **The Guardian Business** | rss | `https://www.theguardian.com/business/rss` | |
| **MarketWatch** | rss | `https://feeds.content.dowjones.io/public/rss/mw_topstories` | |
| **The Economist (Business)** | rss | `https://www.economist.com/business/rss.xml` | Full articles are paywalled; RSS gives headline + summary. Same tier as WSJ links this bot already cites. |
| **Nikkei Asia** | rss | `https://asia.nikkei.com/rss/feed/nar` | **RDF/RSS1.0, not RSS2.0** — feedparser normalizes it the same way as any other feed, but a naive string search for `<item>` (rather than parsing via feedparser) would wrongly read this feed as empty, since RDF uses `<rdf:li>` references instead. Confirmed live during verification. Only source in the registry with an Asia-market angle. |

### Mainstream press — Technology (added 2026-08-13)

| Source | Class | Endpoint | Notes |
|---|---|---|---|
| **BBC Technology** | rss | `http://feeds.bbci.co.uk/news/technology/rss.xml` | |
| **The Guardian Technology** | rss | `https://www.theguardian.com/technology/rss` | |
| **The Economist (Science & Technology)** | rss | `https://www.economist.com/science-and-technology/rss.xml` | Paywalled beyond RSS summary, same as Economist Business. |
| **Wired Business** | rss | `https://www.wired.com/feed/category/business/latest/rss` | |

### Enterprise/industry IT trade press (added 2026-08-13)

| Source | Class | Endpoint | Notes |
|---|---|---|---|
| **The Register** | rss | `https://www.theregister.com/headlines.atom` | Atom, not RSS2.0 — feedparser handles both transparently. 302 redirect, followed automatically. |
| **Computerworld** | rss | `https://www.computerworld.com/index.rss` | 301 redirect, followed automatically. |

### Consumer/gadget tech press (added 2026-08-13)

| Source | Class | Endpoint | Notes |
|---|---|---|---|
| **ZDNet** | rss | `https://www.zdnet.com/rss/all/` | Consumer reviews/how-tos, not industry news — different flavor from the rest of the registry. URL updated 2026-09-03: the old `/news/rss.xml` path is gone entirely — ZDNet moved to a `/rss/<section>/` scheme (confirmed via `https://www.zdnet.com/rssfeeds/`, `/all/` is the general-news one), not a transient outage. Atom, not RSS2.0 — feedparser handles both transparently. |
| **Engadget** | rss | `https://www.engadget.com/rss.xml` | Consumer gadgets/entertainment tech. |
| **TechRadar** | rss | `https://www.techradar.com/rss` (redirects to `/feeds.xml`) | **Blocks the default `python-requests` User-Agent with `403`** — confirmed live. Fixed by sending a self-identifying User-Agent (`_REQUEST_HEADERS` in `news_sources.py`), applied to every source for consistency rather than as a TechRadar-only special case. |


### Added 2026-08-20 — widening away from AI-only feeds

| Source | Class | Endpoint | Notes |
|---|---|---|---|
| **Ars Technica** | rss | `https://feeds.arstechnica.com/arstechnica/index` | Technology, science and policy. The registry had nothing in this register — deeper than the gadget feeds, broader than the AI-only ones. |
| **TechCrunch** | rss | `https://techcrunch.com/feed/` | The **general** feed, deliberately alongside `techcrunch_ai` rather than replacing it. The AI-only one stays; the point of this one is that it isn't AI-only. |
| **CNBC** | rss | `https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114` | Markets, business, economics. Feeds the Finance/Stock categories, which several subscribers' interests map to (AAOI, Bitcoin, 科技財經). |

**Why these three and not a broader news bundle.** The registry was
measured at **28.6% of its cache** coming from feeds that structurally
cannot produce anything but AI content — `openai_blog`,
`huggingface_blog`, `arxiv`, `techcrunch_ai`, `venturebeat_ai` — rising to
**47%** counting `hackernews`, which is heavily AI-skewed in practice. See
`docs/analysis/cluster-measurements.md`.

These three sit inside the product's stated scope
(`agent.LAYER1_IDENTITY`: "a technology industry analyst... covering AI as
well as the broader tech industry"), so they can go straight to
subscribers. General-news feeds (NPR, CBS, CNN, PBS, Politico, The Hill)
would widen the corpus further but mostly produce articles the classifier
tags with nothing, `select_candidate_articles` then filters out, and the
output guardrail would flag — so if they are added later they belong in
`RESTRICTED_SOURCES`, contributing to the corpus and to taxonomy-building
without changing what subscribers receive.

**Breadth, not history.** RSS serves only its current window, so there is
no way to reach backwards — the only lever for a bigger corpus *now*,
rather than after a month of accumulation, is more sources. Each of these
returns 20–30 items per pull against a 200 cap, so the cap is not the
binding constraint here; the number of feeds is.


## Scheduled ingestion pulls by SECTION, not by query (2026-08-21)

This supersedes the query-based descriptions elsewhere in this document
for the four query-capable sources, and only ever applied to
`news_ingest.py`'s scheduled pulls in the first place — `agent.py`'s
`search_news` stopped calling any of these sources live at all once it
was rewired to search the ingested cache instead (2026-09-04, see
`docs/plans/local-news-cache-plan.md` item 5); this section's query-mode-
vs-section-mode distinction is purely a `news_ingest.py` concern now.

| Source | Section endpoint | Sections |
|---|---|---|
| **NewsAPI** | `/v2/top-headlines?category=` | technology, business, science, health |
| **GNews** | `/top-headlines?topic=` | technology, business, science, world |
| **arXiv** | `search_query=cat:` | cs.AI, cs.LG, cs.RO, cs.CR, quant-ph, physics.optics |
| **Hacker News** | `tags=front_page` on `/search` | front_page |

Note the Hacker News endpoint: **`/search`, not `/search_by_date`**, in
section mode only. `front_page` is a ranking, and `search_by_date` would
re-sort it chronologically and discard the only thing it was for. Query
mode still uses `/search_by_date`, as documented above.

**NewsAPI now pins `language=en`**, which GNews always had. Without it
this multilingual aggregator returned whatever matched globally: all 65
cached articles from it were Chinese, against 1 from every other source
combined, and "Bitcoin" returned Spanish-language finance.

### Why this changed

Scheduled pulls used to query these sources with subscriber interest text,
one interest per pull. That makes the corpus a mirror of what subscribers
already asked for: nothing can be discovered that nobody had already
named, and the bias compounds every cycle. It was measurably dirty too —
"AOI" returned Taiwanese optical-inspection news, Japanese anime (AOI is
also a name) and half a page of Chinese.

arXiv's subject classes are the clearest win: free-text search found 36
quantum and 6 optics articles across the whole corpus while subscribers
actively follow both topics. Searching a properly-classified archive by
free text was throwing away its index.

### Consequences worth knowing

**arXiv is uncapped, so all six subject classes are pulled every cycle** —
six calls per 4-hour tick rather than one, a deliberate breadth increase.
Its own multi-day indexing lag prunes most of that before classification.

**Combining the six calls into one OR'd `search_query` was considered
(2026-09-15) and rejected.** arXiv's query syntax does support boolean OR
across `cat:` clauses in a single request, which would cut this to one
call per cycle — but it would reintroduce the exact bug the per-section
cutoff below was built to fix: a single combined request needs a single
shared `since` cutoff and a single shared `max_results`, so cs.AI/cs.LG's
much higher publication volume would both race the cutoff past
quant-ph/physics.optics's rare papers AND crowd them out of the top-N
results entirely — the same "answers to questions nobody asked" kind of
sampling bias this whole section-based design replaced. Fixed the actual
measured problem (429s from a too-short inter-request delay, see the
table above) instead, without touching the six-separate-calls shape.

**The since-cutoff is tracked per `(source, section)`**, not per source.
Sections advance at very different rates — cs.AI produces dozens of papers
a day, physics.optics a handful — and one shared cutoff let the fast
section drag it past the slow one's genuinely new articles, which were then
never offered again. Same class of bug as `last_pulled_at` vs
`last_article_dt`, one level down. Keys are composite (`arxiv:cs.AI`);
sectionless sources keep their plain key, so no migration was needed.

**Perigon has no section vocabulary** — its API has no top-headlines
equivalent — so it falls back to a single fixed query and lost the
rotation it used to have. Accepted rather than overlooked: it has been out
of quota since 2026-08-15 and is excluded from subscriber digests anyway.

## User-Agent

Every fetch sends `Mozilla/5.0 (compatible; ArgusNewsBot/1.0; +https://github.com/nankma/argus)`
via `_REQUEST_HEADERS`. This is a **self-identifying** header, not browser
impersonation — added because TechRadar returns `403` to the bare
`python-requests/x.x` default, and a fake browser UA felt like the wrong
fix for an honest news aggregator. Applied everywhere so a future source
hitting the same block doesn't need its own special case.

## Documented, not yet enabled (need an API key)

Implemented as `news_adapters/*.py` adapter classes, configured via
`news_source.api` entries, but skipped by `enabled_sources()` until the
corresponding env var is set — nothing breaks if it's absent, that
source's entry is just dropped at startup (see
`news_sources._resolved_api_key`). **These are the only three sources in the registry that
would have covered the AAOI gap** (real query-based search across broad
press, including financial press) — getting a free-tier key for any one
of them is a smaller change than adding a new source, since these already
exist in code.

| Source | Class | Env var | Endpoint | Free tier | Notes |
|---|---|---|---|---|---|
| **NewsAPI.org** | api | `NEWSAPI_API_KEY` | `GET https://newsapi.org/v2/everything` | Exists, but "Developer" plan is for testing only, not production; exact rate limit/delay not confirmed on the docs page we checked. Has an explicit `business` category, confirming financial-press coverage. | `q`, `sortBy=publishedAt`, `pageSize`, `apiKey` (or `X-Api-Key` header). |
| **GNews** | api | `GNEWS_API_KEY` | `GET https://gnews.io/api/v4/search` | 100 requests/day, 10 articles/request, 1 req/sec, **12-hour delay on articles**, non-commercial only. Resets 00:00 UTC. | `q`, `lang`, `max`, `apikey`. |
| **Perigon** | api | `PERIGON_API_KEY` | `GET https://api.perigon.io/v1/all` | 150 requests/month, non-commercial only. | Response field names (`source.domain`, `summary`, `pubDate`) taken from general docs, **not verified live** — no key to test with. Double-check against a real response before relying on it. |

None of these were live-tested (no credentials available). Endpoint shapes
came from each provider's docs — verify against a real response the first
time a key is actually configured, in case something's drifted.

### Getting a key — recommendation and order

**Start with GNews, not NewsAPI, despite NewsAPI being the more
feature-rich and better-known of the two.** The deciding factor is ToS
fit, not features:

| | Free-tier restriction | Fit for this project |
|---|---|---|
| **GNews** | "Non-commercial use only" | Fits — this is an unpaid pilot, invite-gated, nothing sold. The 12-hour article delay is a real cost, but acceptable for the intended use (an on-demand fallback for specific/low-profile queries like AAOI, not the primary real-time push feed). |
| **NewsAPI** | Free "Developer" plan is **explicitly for development/testing only, not production** | Doesn't fit cleanly — this bot serves real subscribers, which is production use by any reasonable reading. This project has already turned down otherwise-working sources on similar grounds (Reddit's blocked endpoint, Google News's link-resolution issue) rather than use something in a way its provider didn't intend. Worth revisiting only if paying for a real plan is later on the table. |
| **Perigon** | 150 requests/month, non-commercial only | Technically fits the ToS, but 150/month is too low for routine use across multiple subscribers — viable only as an occasional supplementary source, not a primary one. Lowest priority of the three. |

**Sign-up steps for GNews:**

1. Go to `https://gnews.io/register` and create a free account (email + password, no payment method required for the free tier).
2. After registering, the API key is shown directly on the account dashboard (`https://gnews.io/dashboard`) — no separate approval step.
3. Send the key value in this conversation (or set it directly as an env var if working locally) and it'll be wired in: locally via `$env:GNEWS_API_KEY = "..."` for testing, and into OCI Vault following the existing secrets pattern (`docs/plans/security-plan.md` finding 2) for the deployed bot — same handling as `DEEPSEEK_API_KEY`/`TELEGRAM_BOT_TOKEN`, never as a plaintext env var in the running container.
4. Once the key is set, `enabled_sources()` picks it up automatically — no code change needed, it's already implemented and registered.

**After GNews is working and verified live** (confirm the response shape actually matches what `GNewsAdapter.pull` expects — per this doc's standing rule, verify before trusting), Perigon is the natural second pick if broader coverage is still wanted. NewsAPI stays parked unless a paid plan is actually being considered.

## Restricted sources: NewsAPI and Perigon require per-user access

Added 2026-08-14, after realizing `search_news` (the on-demand chat tool,
as it worked at the time) called every enabled source directly and live,
on every matching query, completely independent of `news_ingest.py`'s
own budget-cap mechanism (see `docs/plans/local-news-cache-plan.md`).
That mechanism only protected the periodic ingestion job's own calls —
nothing then stopped `search_news` from also calling NewsAPI/Perigon on
every relevant chat message, which would exhaust both budgets almost
immediately on real traffic. **`search_news` no longer calls any source
live at all** (rewired 2026-09-04 to search the ingested cache instead —
`docs/plans/local-news-cache-plan.md` item 5), so this restriction now
applies only to `news_ingest.py`'s own scheduled pulls and
`news_push.py`'s digest filtering, described below as it still works
today.

**`news_sources.RESTRICTED_SOURCES = {"newsapi", "perigon"}`** — excluded
from `news_ingest.py`'s scheduled pulls and `news_push.select_candidate_articles`'s
digest filtering by default, both keyed off the same per-user flag,
`subscriber_ops.get_restricted_sources_enabled(chat_id)` (defaulting to
`False`). `bot.py`/`combined_bot.py` grant this to the admin's own
chat_id at startup — nobody else, for now. Granting it to someone else
later is a plain DB update (`subscriber_ops.set_restricted_sources_enabled(chat_id,
True)`), not a new code path.

**GNews is deliberately not restricted** — its 100/day budget has real
headroom beyond what `news_ingest.py` alone uses (3–6 calls/day).

**What this does and doesn't solve.** It protects Perigon/NewsAPI's
budgets from *unauthorized* scheduled-ingestion usage — the default is
"nobody but the admin's own interests pull from these sources." Since
`search_news` stopped calling sources live entirely, the earlier concern
here (the admin's own on-demand usage not being rate-limited against
`news_ingest.py`'s own consumption of the same cap) no longer applies —
there is no on-demand call left to rate-limit against it.

**Real gap found and fixed 2026-08-14**: this restriction was only ever
applied to `search_news` (the on-demand chat tool, as it worked at the
time). `news_push.py`'s periodic-digest cycle called
`news_sources.enabled_sources()` with no argument at all, which defaults
to `include_restricted=True` — so every push-enabled subscriber's digest
fetch included NewsAPI/Perigon regardless of their own
`restricted_sources_enabled` flag, the exact thing this section says is
supposed to default to "nobody but the admin." Found while diagnosing a
real subscriber's stalled pushes (a separate, unrelated `TypeError` in
`_parse_iso_published` was the actual crash — see that function's
docstring — but this gap meant restricted sources were live in every
subscriber's push path either way). Fixed by adding
`list_push_enabled_subscribers`' `restricted_sources_enabled`
field and threading it through `run_push_cycle`, whose own default was
flipped to `False` (unlike `enabled_sources` itself) so a future caller
that forgets to pass it explicitly fails closed, not open. **Superseded
the next day (2026-08-15)** when `news_push.py` stopped calling
`news_sources` live at all and converged onto the local cache (see
`docs/plans/local-news-cache-plan.md`'s "Interaction with `news_push.py`") —
the same gating now lives in `news_push.select_candidate_articles`,
checking each cached article's `source_key` against `RESTRICTED_SOURCES`
instead of gating a live source list, same effect.

## Considered, tested live, and rejected

All verified live on 2026-08-13 before being ruled out — per this doc's
own standing rule (see "How to add a new source" below), nothing here is
rejected on a docs page's word alone.

| Source | Why not |
|---|---|
| **CNN** (`rss.cnn.com/rss/cnn_tech.rss`, `money_latest.rss`) | Feed responds `200`, but `lastBuildDate` is over a year stale — the infrastructure is abandoned, not actively publishing. Their newer `edition.cnn.com/business/rss` path returns `404`; no working replacement found. |
| **CNBC** (Technology and Finance sections) | `403 Forbidden` on both, even with a browser-like User-Agent. Blocked at a level beyond what a UA header fixes. |
| **Fortune** | `403 Forbidden`. |
| **Reuters** | `404` — discontinued public RSS in 2020 (previously documented; reconfirmed live this round). |
| **Business Insider** (default `/rss`) | Feed is live, but it's their general firehose (politics, military, world news) — not business/tech-specific despite the section name. Their `/tech/rss` and `/business/rss` paths both return `404`. |
| **InfoWorld** | `404`. |
| **Yahoo Finance** (per-ticker RSS, e.g. `finance.yahoo.com/rss/headline?s=AAOI`) | `429` — rate-limited/deprecated. |
| **Seeking Alpha** (per-symbol RSS, e.g. `seekingalpha.com/symbol/AAOI.xml`) | `403` — blocked. |
| **Nikkei Asia's plain feed URL grepped for `<item>`** | Not a rejection of the source (it's enabled — see above), but a methodology trap worth recording: naively grepping for `<item>` on this feed finds zero, because it's RDF/RSS1.0. Always verify via `feedparser`, matching how the code actually parses it, not a raw string search. |
| **Google News RSS search** (`news.google.com/rss/search?q=...`) | Works, and covers almost anything (validated: real AAOI coverage, and 100 results for a generic "NVIDIA chip" query from CNBC/Bloomberg/Financial Times/Benzinga). **Rejected anyway**: its `<link>` doesn't resolve to the actual article via a normal HTTP fetch — `curl -L` lands on a Google interstitial page that needs client-side JavaScript to redirect further to the real publisher URL. That breaks this project's requirement that `search_news` return real, citable URLs (see `CLAUDE.md`'s note on why `link` was added to `search_news`'s output in the first place). A source that regresses that isn't worth adding even though its coverage is excellent. |
| **Reddit** (e.g. r/MachineLearning, r/LocalLLaMA) | Reddit deprecated unauthenticated `.json` endpoint access around 2026-05-28 — a request with a proper custom `User-Agent` still returns `403 Forbidden`. Getting Reddit data now requires registering an OAuth app (`reddit.com/prefs/apps`) and using the official API — more setup than a simple API key, not done yet. |

## How to add a new source

**A plain RSS/Atom feed (query-less, no API key)** — no code change:

1. Add an entry to `settings.yml`'s `news_source.rss` list: `{key: your_key, display_name: "Source Name", url: "https://.../feed.xml"}`. `news_sources._rss_sources_from_settings()` picks it up automatically as `enabled_sources()`'s next call.
2. **Also add the exact same entry to `settings.oracle.yml` and `settings.int.yml`** — production and INT each read their own file, never `settings.yml`; an edit only there is invisible to a deployed instance (see `docs/standaloneplan/01-settings-migration.md`'s "Migration methodology" for why this atomicity matters, and `settings.yml`'s own `news_source.rss` comment).
3. Add a row to the appropriate table above.
4. **Test it live before trusting it** — several entries in this doc exist because a docs page or search summary turned out to be wrong (see `CLAUDE.md` for the DeepSeek-model-retirement false alarm and the OK Surf News API response-shape mismatch from earlier in this project's history). Check the HTTP status, the actual item count via `feedparser` (not a raw string search — see the Nikkei Asia note above), and how stale `lastBuildDate` is before adding it. `news_sources._make_rss_fetcher(url, "Source Name")()` in a throwaway Python shell is the quickest way to check.

**A source with real per-source logic** (a query parameter, a date-range filter, an API key, pagination — anything `_fetch_rss(url, name, max_results)` alone can't express) needs a `NewsSourceAdapter` class under `news_adapters/`, not a hardcoded registry entry:

1. Write `news_adapters/<name>.py` declaring a class with `TYPE = "<name>"`, `initialize(self, config: dict) -> None` (store `config["api-key"]` if the source needs one; a no-op otherwise), and `pull(self, query: str, max_results: int, since: datetime | None = None, section: str | None = None) -> list[dict]`, returning a list of `{"title", "link", "source", "summary", "published", "published_dt"}` dicts (any field can be `None` if the source doesn't provide it). `since`/`section` must both be accepted even if the adapter ignores them (see `PerigonAdapter` — no `since`, no `section`), to satisfy the Protocol shape uniformly. `news_sources.discover_adapter_types()` picks the class up automatically at process startup by scanning `news_adapters/` for classes with a `TYPE` attribute — no registry to edit by hand.
2. If it needs an API key, add an entry to `news_source.api` in `settings.oracle.yml` (live, if production has the key) and `settings.yml` (commented, as a preview otherwise): `{key: <name>, type: <name>, api-key: {trailsign-resolve: environment-variable, name: YOUR_ENV_VAR_NAME}}` — **never include the block in a file where the underlying env var genuinely won't be set**; an entry whose credential can't resolve is silently dropped (that source stays off), not a crash (see `news_sources._raw_api_entries`/`_resolved_api_key`'s own docstrings), but a `type` with no matching adapter class under `news_adapters/` fails the whole process at startup (`validate_configured_types`) — ship the adapter file first.
3. If it needs no credential and should always be on (like `hackernews`/`arxiv`), wire it in directly via `news_sources._always_on_sources()` instead of a `news_source.api` entry — there's nothing to configure for a source with no credential and no override.
4. Add a row to the appropriate table above.
5. Test it live before trusting it, same as step 4 above.
