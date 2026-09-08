# analysis/

Research and measurement for **how news gets ranked and how we find out
what a subscriber wants** — what makes an article important, how to
work out what someone is actually interested in, and how many items a
digest should carry.

Separate from `docs/plans/` on purpose. Those documents track *what we
decided and built*. These track *what we measured and what the literature
says*, much of which will never become code. They also carry the raw
numbers behind a decision, which would bury a plan doc.

## What's here

| Document | What it is |
|---|---|
| [`news-ranking-plan.md`](news-ranking-plan.md) | The main survey: how importance gets defined (journalism news values, computational signals, LLM-as-judge), how user preference gets matched, how digest size gets decided. Five concrete options with cost/risk. Also carries the 2026-08-18 source-collapse diagnosis |
| [`sample-diversity-survey.md`](sample-diversity-survey.md) | Cross-domain companion: how statistics, ecology, finance, clinical trials, astronomy, genomics, IR/ML and search engines each handle an over-concentrated sample — and the four different pipeline stages they intervene at |
| [`cluster-measurements.md`](cluster-measurements.md) | The measured numbers: how many story clusters the real cache contains, how big they are, and five findings that came out of measuring rather than assuming. **Links to the interactive scatter plots** |
| [`interest-elicitation-survey.md`](interest-elicitation-survey.md) | Cross-domain survey for a proposed "help me find my interests" conversation: what IR theory, career-interest inventories, journalism/clinical interviewing, personal construct psychology, motivational interviewing, conversational recommenders and onboarding UX each do about the same problem — plus what this codebase already has, and the constraints its own measured history imposes |

`news-ranking-plan.zh.md` and `sample-diversity-survey.zh.md` are Chinese
translations. **Keep each pair in sync** — a translation that drifts from
its original is worse than no translation.

Mostly not built — these documents inform decisions that haven't been
made yet. The one exception is `interest-elicitation-survey.md`, whose
findings did become code: see `docs/plans/interest-finder-plan.md` for
which three of them shaped it and how.

### Documents referenced from here

These documents cite others by bare filename. They live elsewhere in the
repo:

| Referenced as | Actually at |
|---|---|
| `system-overview.md` | [`docs/system-overview.md`](../system-overview.md) — architecture, and Appendix B.1's measurement discipline |
| `ai-news-sources.md` | [`docs/current/ai-news-sources.md`](../current/ai-news-sources.md) — the source registry and per-source classes |
| `local-news-cache-plan.md` | [`docs/plans/local-news-cache-plan.md`](../plans/local-news-cache-plan.md) — the cache, and the cap raise that exposed source collapse |
| `model-portability-plan.md` | [`docs/plans/model-portability-plan.md`](../plans/model-portability-plan.md) — the "measure before shipping" precedent |

## Tools

Three scripts, run in order. Only the first needs the production VM;
everything after it works offline on the saved snapshot, so the analysis is
free to re-run.

All computation is local — **scikit-learn only, no API calls, no LLM calls,
nothing metered**. Unlike `tools/measure_guardrails.py` and
`tools/run_eval.py` at the repo root, these cost nothing to run.

### 1. `fetch_cache_snapshot.py` — pull the cache off the VM

```bash
python docs/analysis/tools/fetch_cache_snapshot.py \
    --host ubuntu@<bot-vm-ip> \
    --key "C:/Users/<you>/path/to/ssh-key.pri.key"
```

Writes `analysis/data/cache-snapshot.tsv` (one row per cached article:
source, publish date, title, summary).

VM address and key path are in `local-infra/infrastructure.yaml`, which is
gitignored — read them from there, don't hardcode.

> The bot VM has no Python on the host and none inside the container, so
> this shells out to `grep`/`cut` over the YAML cache files rather than
> running a script remotely.

### 2. `cluster_news.py` — how many clusters, and how big

```bash
python docs/analysis/tools/cluster_news.py                 # the sweep table
python docs/analysis/tools/cluster_news.py --detail        # every cluster's contents
python docs/analysis/tools/cluster_news.py --json out.json # machine-readable
```

Clusters with both TF-IDF cosine and BM25 across a threshold sweep, and
reports cluster counts, singleton counts, cross-source counts and size
histograms. Also counts exact-duplicate cache entries.

`--detail` is the one worth running when a number looks surprising — it
prints what is actually inside each multi-article cluster, which is how the
aggregator-echo and same-source-series false positives were found.

### 3. `build_cluster_report.py` — the visual report

```bash
python docs/analysis/tools/build_cluster_report.py
python docs/analysis/tools/build_cluster_report.py --highlight-source gnews
```

Projects the cache to 2D (TF-IDF → SVD(50) → t-SNE) and writes
`analysis/cluster-report.html` — one self-contained file, no external
assets, opens straight from disk.

Two scatter panels: story clusters (singleton / same-source / cross-source),
and one source against all others. `--highlight-source` picks which source
the second panel isolates; it defaults to `hackernews` because that's the
one that dominated the digest.

Takes a minute or two — t-SNE is the slow part.

> **The HTML is generated, never hand-edited**, and is gitignored for the
> same reason as `showcase.html`: it's a ~700 KB build artifact derived from
> a snapshot that goes stale within days. The findings that outlive it live
> in `cluster-measurements.md`.

## Requirements

Everything is already in `environment.yml` — `scikit-learn`, `numpy`,
`scipy`. Nothing extra to install:

```powershell
conda activate myfirstagent
```

Deliberately **not** used:

- `rank_bm25` — BM25 is a short formula, implemented inline rather than
  adding a dependency for one experiment.
- `sentence-transformers` — dense embeddings would need it, and it pulls in
  PyTorch (~2 GB). Whether to take that on is an open decision, tracked as
  Option E tier 2 in `news-ranking-plan.md`.
- `matplotlib` — the report renders SVG directly, so there's no plotting
  dependency and the output is interactive rather than a static image.
