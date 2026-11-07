# Cluster Measurements — the real cache, 2026-08-19

The numbers behind `news-ranking-plan.md`'s Option B. Everything here was
measured on a live snapshot of the production cache, not estimated.

**📊 Interactive report with both scatter plots:**
[cluster-report.html](cluster-report.html) — open it locally, or view the
published copy at
<https://claude.ai/code/artifact/c8b4c66f-ce7d-480f-80b6-d6bc3bc49ef7>.

The HTML is **generated, never hand-edited** — regenerate it any time with
`python docs/analysis/tools/build_cluster_report.py` (see [README](README.md)).
It is gitignored, like `showcase.html`, because it is a 700 KB build
artifact derived from a snapshot that goes stale within days. This file
holds the findings that outlive it.

---

## Snapshot

| | |
|---|---|
| Articles | 2,082 |
| Sources | 21 registered, 20 present in the window |
| Window | 48 h (the cache TTL) |
| With summary text | 1,469 of 2,082 |
| Snapshot taken | 2026-08-19 |

## Story clustering — how many clusters, and how big

Connected components (single-linkage) over a similarity threshold, which is
the standard approach in the near-duplicate literature: if A~B and B~C then
A, B and C are one story.

### TF-IDF cosine

| Threshold | Clusters | Singletons | Multi-article | Cross-source | Largest |
|---|---|---|---|---|---|
| 0.30 | 1,985 | 1,913 | 72 | 15 | 5 |
| **0.40** | **2,028** | **1,988** | **40** | **7** | **5** |
| 0.50 | 2,044 | 2,015 | 29 | 6 | 5 |
| 0.60 | 2,053 | 2,032 | 21 | 3 | 5 |
| 0.70 | 2,065 | 2,049 | 16 | 0 | 3 |

### BM25 (normalized)

| Threshold | Clusters | Singletons | Multi-article | Cross-source | Largest |
|---|---|---|---|---|---|
| 0.10 | 117 | 103 | 14 | 7 | **1,951** ⚠ |
| 0.15 | 1,060 | 865 | 195 | 106 | **651** ⚠ |
| 0.20 | 1,714 | 1,535 | 179 | 70 | 17 |
| 0.30 | 1,976 | 1,910 | 66 | 22 | 8 |
| 0.40 | 2,033 | 1,994 | 39 | 9 | 5 |

⚠ = degenerate. The clustering has collapsed, not converged.

### Size distribution at the working point (TF-IDF 0.40)

| Cluster size | Count |
|---|---|
| 1 (singleton) | 1,988 |
| 2 | 32 |
| 3 | 3 |
| 4 | 4 |
| 5 | 1 |

**2,082 articles → 2,028 clusters is a reduction of 54, or 2.6%.** Only 94
articles join a cluster at all. Clustering barely consolidates anything —
that is the finding, not a caveat to it.

## Findings

### 1. The corroboration signal is real but very thin

Seven cross-source clusters in a 48-hour window. Enough to identify the
handful of stories the whole industry covered at once, and nothing more.
**Corroboration can promote the top few items; it cannot rank the other
~2,000**, which all score identically as uncorroborated. Option B needs a
second signal underneath it.

### 2. BM25 collapses catastrophically below ~0.20

At threshold 0.10, **1,951 of 2,082 articles land in one cluster**. This is
single-linkage chaining — A resembles B, B resembles C, so all three merge,
repeated until nearly the whole cache is one story. Not a BM25 defect; it is
what connected-components clustering does with a permissive threshold on a
dense similarity matrix. TF-IDF cosine happens to be sparse enough that it
never gets there. **If BM25 is used, the threshold has to be validated
against a real snapshot, not assumed.**

### 3. There is no dense cluster structure to find

Compressing the article vectors to 50 dimensions retains only **6.6% of the
variance**. 2,082 articles from 20 outlets over 48 hours are genuinely about
~2,000 different things. This is why the scatter plot is a diffuse cloud
rather than tidy blobs, and why k-means (K=12, tried separately) put 59% of
everything into a single catch-all cluster.

### 4. The dominant source overlaps everything — which explains the MMR result

Hacker News (360 articles, the most of any source) does not occupy its own
region of content space:

| | |
|---|---|
| HN centroid → all-other-sources centroid | 0.175 |
| HN's own spread | 0.205 |

**The separation is smaller than the spread** — the two distributions sit on
top of one another. This is the picture behind the separately-measured
negative result that MMR on content similarity left source concentration
unchanged (effective sources 3.1 → 3.1). Content-based diversity works by
pushing apart points that sit close together, and HN's articles are already
spread across the whole space. It dominates by publishing *frequently*, not
*repetitively*, and no similarity function can see the difference.

### 5. Four kinds of thing end up in a "cluster" — only one is wanted

Inspecting all 40 multi-article clusters at TF-IDF 0.40:

- **Genuine cross-source corroboration** — Apple's camera-equipped AirPods
  across Engadget, TechCrunch and HN; Google buying Spirit Airlines' data
  across GNews and HN; Sonos lock-screen controls across Engadget and
  TechRadar. *This is the signal.*
- **Aggregator echoes** — an HN entry linking to a BBC article carries BBC's
  exact headline. One article counted twice, inflating the source that
  already dominates.
- **Same-source series** — OpenAI's blog posting "Sales workflows with
  ChatGPT Work", "Data science workflows with ChatGPT Work", and so on.
  Lexically near-identical, editorially four different posts. A false
  positive.
- **Literal duplicates** — 14 groups, 19 redundant copies, mostly recurring
  RSS programme slots ("Tech Now", "Tech Life", "Business Daily") that reuse
  one title for every episode. Invisible to `news_cache`'s link-hash dedup
  because the links differ.

**Any corroboration count must collapse aggregator echoes onto their origin
first**, or it will systematically over-credit the most over-represented
source.

## Reproducing this

See [README.md](README.md). Three commands, and only the first needs the VM:

```
python docs/analysis/tools/fetch_cache_snapshot.py --host ubuntu@<ip> --key <key>
python docs/analysis/tools/cluster_news.py
python docs/analysis/tools/build_cluster_report.py
```

All computation is local — scikit-learn only, no API calls, no LLM calls,
nothing to pay for. The full pairwise similarity matrix over 2,082 articles
takes about a second and peaks around 36 MB.

## Caveats

- **One snapshot, one 48-hour window.** Everything here could look different
  during a major news event, when corroboration would spike. Worth re-running
  on a busy day before treating these numbers as typical.
- **t-SNE layout is not metric.** Distances between distant points in the
  scatter carry no meaning; only local neighbourhoods do. The centroid and
  spread figures in finding 4 are computed in that projected space and are
  best read as *comparative* (separation vs spread), not absolute.
- **Titles and short summaries only.** The cache does not hold full article
  text, so clustering sees roughly a headline plus up to 300 characters.
  Full text would likely find more corroboration.

---

# Follow-up, 2026-08-19: clustering the *filtered* set, and can clusters replace the 13 categories?

Two questions raised after seeing the scatter plot, plus an outage found
while answering them.

## First: the 13 categories are not working at all

Before either question could be answered honestly, this had to be
established. Of 2,262 cached articles, **only 163 (7.2%) carry any
category**. The other 92.8% are uncategorized — and
`news_push.select_candidate_articles` excludes uncategorized articles
whenever the subscriber's topic has real categories, so most of the cache
is invisible to most subscribers.

Cause, found by grouping cached articles by the ingestion tick that wrote
them:

| Batch size | Categorized |
|---|---|
| 1 | 100% |
| 60 | 95% |
| 109 | 96% |
| **113** | **0%** |
| 139 / 147 / 156 / 171 | **0%** |
| 281 | **0%** |
| 1085 | **0%** |

All-or-nothing, with a cliff just above 110. `news_classify` made **one
call per ingestion cycle**, which was fine while a cycle produced ~100
articles and stopped being fine when `local-news-cache-plan.md` item 7
raised the RSS per-source cap from 5 to 200 and cycles began producing
100–1000+. The likely mechanism is the structured-output response
exceeding the model's output token limit — one entry per article, so
response length scales with batch size.

It ran for three days without a trace, because `classify_articles` caught
every exception and returned `{}` with no logging. **Fixed**: batches are
chunked at `MAX_ARTICLES_PER_CALL = 50`, a failure now prints, and a
failed chunk costs one chunk instead of the whole cycle.

A second, compounding failure in the same area: several cached
interest→category mappings are empty (`AI` → `[]`, `robotics` → `[]`,
`Bitcoin` → `[]`, `機器人科技` → `[]`). An empty mapping is treated as
unrestricted by design, so those subscribers match *every* article, while
subscribers whose interests did map correctly match almost none.

## Q1 — What do the filtered results cluster into?

Traced for a real subscriber (interests: semiconductors, quantum
computing, robotics):

| Interest | Mapped to | Candidates |
|---|---|---|
| semiconductors | `Hardware` | 32 of 2,262 |
| quantum computing | `Research`, `Hardware` | 42 of 2,262 |
| robotics | *(empty mapping)* | **2,262 — no filtering at all** |

Clustering the 32 `Hardware` articles gives **31 clusters: 30 singletons
and one pair**. There is nothing to cluster.

**The honest answer to Q1 is that the filtered set can't be meaningfully
clustered in its current state** — not because clustering doesn't work,
but because the filter is either returning ~1% of the cache or 100% of it.
Worth re-running once classification is repaired and the cache has
refilled with properly categorized articles.

## Q2 — Could the visual clusters replace the 13 categories?

The intuition is well-founded. Formalizing the hand-circled clumps with
HDBSCAN over the t-SNE layout finds **~80 clusters covering 89% of
articles**, and they are far more coherent than the hand-written
categories when measured back in the original TF-IDF space:

| | Median coherence | vs random baseline |
|---|---|---|
| Discovered clusters | 0.0869 | **9×** |
| The 13 categories | 0.0241 | 3× |

They also read as genuinely specific topics — `quantum computing` (47
articles), `apple / camera / airpods` (30), `bitcoin / price / btc` (68),
`google / pixel` (50), `data center` (41), `supply chain` (18) — where the
hand-written set offers only `Hardware` or `Finance`. Two categories,
`Hardware` and `Policy`, score at **1× baseline**: articles labelled with
them are no more similar to each other than two random articles.

**But they cannot be categories, for one decisive reason: they are not
stable.** Re-running the identical pipeline on the identical data with
only the random seed changed:

| | Run A | Run B |
|---|---|---|
| Clusters | 77 | 84 |
| Noise points | 255 | 213 |
| **Adjusted Rand Index between the two groupings** | &nbsp; | **0.478** |

Less than half the grouping survives a change of seed *on the same data*.
Real data changes every four hours. A category has to be a stable
vocabulary that a stored user interest maps onto once and keeps matching;
"cluster 36" means something different on the next run, so there is
nothing durable to store a preference against.

Two further caveats: several clusters are **single-source artifacts** (one
outlet's beat, not a topic — the 68-article bitcoin cluster is one
source), and the tail includes incoherent grab-bags (a 94-article cluster
whose top terms are `null, v0, earth, dream, transformer`).

### Conclusion

**Not a replacement — a complement, at a different layer.**

- **Categories** stay the stable, coarse vocabulary that a *stored user
  interest* matches against. Their job is durability across time, and
  clusters cannot do it. What they need is to actually run (see the outage
  above) and probably a better taxonomy — `Hardware` and `Policy` earning
  1× baseline is evidence the current 13 are partly arbitrary.
- **Clusters** are valuable *within a single push cycle*, where stability
  across runs is irrelevant: grouping the current candidate set by story
  for de-duplication, for source diversity, and for structuring the digest
  itself. That is exactly what `news-ranking-plan.md`'s Option B needs, and
  the 9× coherence says the signal is real.

The measurement that would change this conclusion: if cluster assignments
proved stable across *consecutive real ingestion cycles* (not just seeds),
the durability objection weakens. Worth testing before dismissing the idea
permanently — the coherence numbers are strong enough to justify it.

---

# Follow-up, 2026-08-19: picking an embedding backend, and whether it fits the VM

Everything above used TF-IDF as a stand-in for "embeddings", because the
project had no embedding model. That stand-in is what produced the two
headline failures — word-attractor clusters, and a degenerate distance
metric where 1,382 of 2,257 candidate articles tied at exactly zero
distance from the selected set. So before designing anything on top of
clusters, the question is which backend to actually use.

Measured with `docs/analysis/tools/bench_embeddings.py` on the same 2,262
article snapshot. Each backend runs in its **own process** — RSS is
process-wide and `del` + `gc.collect()` does not return pages to the OS,
so measuring them in one process credits whichever runs last with
everything before it. The first version of that script did exactly that
and reported fastembed at +1,077 MB.

## The numbers

| backend | dim | disk | RSS peak | docs/s | separation | med max-sim | exact zeros |
|---|---|---|---|---|---|---|---|
| TF-IDF (incumbent) | 4494 | 0 MB | **9 MB** | 43281 | **−0.101** | 0.0000 | 1382/2257 |
| model2vec potion-base-8M | 256 | 8 MB | **80 MB** | 22145 | +0.155 | 0.2349 | 6/2257 |
| fastembed bge-small (ONNX) | 384 | 54 MB | 855 MB | 8 | +0.195 | 0.5853 | 0/2257 |
| sentence-transformers MiniLM-L6 | 384 | 634 MB | 416 MB | 129 | **+0.299** | 0.1752 | 11/2257 |

**"Separation"** is the gap between the lowest-scoring pair that *should*
match and the highest-scoring pair that shouldn't, across four probes.
Absolute cosine values are not comparable across backends — each model
family has its own similarity floor, and BGE scores everything high — so
the gap is the comparable quantity.

**TF-IDF's separation is negative.** Its false positive (two articles
sharing only the word "google", 0.101) outscores a true match (same topic,
different vocabulary, 0.000). On this task TF-IDF is not merely weak, it
is anti-correlated. That is the single most useful number here: it retires
the "maybe TF-IDF is good enough" option outright.

## Does it fit the bot VM?

Measured on the live box: `VM.Standard.E2.1.Micro`, 954 MB total, **420 MB
available** with `myfirstagent-bot` already resident at 184 MB, 1 OCPU,
x86_64, 33 GB free disk, 1 GB swap of which 240 MB is already in use.

| backend | verdict |
|---|---|
| model2vec | **fits** — 80 MB, 340 MB headroom |
| sentence-transformers | **does not fit in practice** — 416 MB peak against 420 MB available is 4 MB of headroom, plus 634 MB added to the Docker image |
| fastembed | **does not fit** — over by 435 MB, and 8 docs/s makes it moot anyway |

Disk is not the constraint. Peak RSS is.

## Frozen-taxonomy quality on dense vectors

Rerun of the frozen-taxonomy experiment (build clusters on the older 60%,
assign the unseen newer 40%) with each backend:

| backend | clusters | coverage, all | coverage, pruned | median best-match vs random |
|---|---|---|---|---|
| TF-IDF | 55 | 82.5% | **43.3%** | 0.216 vs 0.006 |
| model2vec | 35 | 89.4% | **80.8%** | 0.446 vs 0.186 |
| sentence-transformers | 36 | 86.0% | **75.8%** | 0.449 vs 0.143 |

The pruning column is the important one. TF-IDF needed aggressive pruning
because half its clusters were word-attractors, and pruning cost it 39
points of coverage while contamination survived anyway. Dense vectors
produce fewer junk clusters to begin with (35–36 rather than 55, worst
coherence 1.5× baseline rather than near-random), so pruning costs 9–10
points instead of 39.

The specific failure case is fixed in both: "EmbeddingGemma, Google's new
efficient embedding model" no longer lands in a Spirit Airlines cluster.

At tight thresholds sentence-transformers pulls ahead (32.6% vs 20.2% at
sim ≥ 0.50), which is consistent with its better separation. model2vec
buys ~85% of the useful behaviour for 19% of the RAM, no torch dependency,
and 87× faster encoding (0.2 s vs 17.4 s for the full corpus).

## Reproducing

```powershell
conda activate myfirstagent
python docs/analysis/tools/bench_embeddings.py
```

## Correction: peak-RSS was measuring the wrong workload

The table above measured peak RSS while batch-encoding all 2,262 articles.
That is not the production workload. The intended split is:

1. Build the taxonomy **offline on a dev machine** — encode the corpus,
   cluster, prune. Memory is free here.
2. Store the resulting centroids. A 35-cluster taxonomy is **54 KB**
   (35 × 384 float32) — trivially a DB row or a checked-in file.
3. On the VM, encode **each incoming article** and take the nearest
   centroid. That is a 384-vector against a 35 × 384 matrix.

Step 3 is genuinely light *in work*. It is not light *in residency*, and
that distinction is what actually decides the backend: the model has to be
loaded in RAM to encode anything at all, and that fixed cost does not
shrink with batch size. Re-measured with the two costs separated:

| backend | FIXED (import + load) | MARGINAL (all encoding) | total resident | per article | separation |
|---|---|---|---|---|---|
| model2vec | **84 MB** (1.1 s) | 7 MB | **91 MB** | **0.2 ms** | +0.155 |
| fastembed bge-small | **152 MB** (0.5 s) | 20 MB | **172 MB** | 96–300 ms | +0.195 |
| sentence-transformers | **398 MB** (14.6 s) | 90 MB | **488 MB** | 11–21 ms | +0.299 |

Against the VM's 420 MB available: model2vec leaves 329 MB, fastembed
leaves 248 MB, sentence-transformers overruns by 68 MB — and would overrun
on the fixed cost alone if anything else on the box moved.

**The earlier fastembed number was wrong.** It was recorded at 855 MB peak
and 8 docs/s, and both figures were artifacts of `TextEmbedding.embed()`
forking parallel workers by default. Pinned to `threads=1, parallel=0`, its
fixed cost is 152 MB — a factor of 5.6 lower, and the difference between
"disqualified" and "viable". Worth flagging as a measurement lesson: a
default that spawns workers will misreport both memory and throughput on a
single-core target.

fastembed is nonetheless the **slowest** per article of the three, ONNX and
int8 notwithstanding — 96–300 ms against sentence-transformers' 11–21 ms,
because single-threaded int8 gets no benefit from the BLAS paths torch
uses. That is tolerable for a background ingestion job (200 articles ≈ 19 s
here, more on 1 OCPU) and irrelevant to user-facing latency, but it rules
fastembed out of anything synchronous.

**Two viable candidates**, then: model2vec (cheapest, fastest, weakest
separation) and fastembed (2× the memory, ~1000× the latency, better
separation). sentence-transformers stays the offline quality reference and
cannot be deployed to this VM.

---

# Rough taxonomy from embeddings, both candidates, 2026-08-19

Built with `docs/analysis/tools/build_taxonomy.py` on the same 2,262
article snapshot. The pipeline is the three-step design: encode and
cluster offline, keep the centroids, classify new articles against them at
runtime with no LLM call.

Cluster labels come from **c-TF-IDF** — pool each cluster's documents into
one pseudo-document, then TF-IDF across clusters, so terms frequent *in* a
cluster but rare *across* clusters win. Plain centroid-nearest-terms gave
labels like "ai, new, hn" because it just surfaces whatever is globally
common.

## The finding that unblocked fastembed: anisotropy

Run as-is, fastembed produced **2 clusters and 2 noise points from 2,262
articles** — one undifferentiated blob. The cause is visible in one number:
its **random-pair cosine baseline is 0.502**. Two unrelated articles are
already 50% similar, so a density-based clusterer has no contrast to grip.
model2vec's baseline is 0.120.

This is the well-known anisotropy of BERT-family embeddings — they occupy a
narrow cone rather than filling the sphere. Subtracting the corpus mean
before normalizing fixes it completely: baseline 0.502 → 0.001, and 2 raw
clusters → 75.

**Centering is not optional for either backend, and it must use the build
set's mean, not the whole corpus's.** Getting that wrong is subtle and was
a real bug here: centering the build subset by a mean computed over a
superset leaves a residual offset that re-introduces the anisotropy, and it
collapsed the model2vec holdout run from 40 clusters to 2. It also matters
practically — at build time only the build set exists. The mean is stored
with the centroids and must be applied to every article classified later,
or runtime vectors land in a different space than the taxonomy.

## The derived categories

Both produce plausible topic sets, but with a consistent difference in
grain. Selected, by size:

| model2vec (36 kept) | fastembed (38 kept) |
|---|---|
| bitcoin, btc, price | bitcoin, price |
| chatgpt, chatgpt work, codex | openai, safety, frontier |
| cyber, cybersecurity | codex, openai, gpt |
| unitree, robot, humanoid | unitree, humanoid, robot |
| hugging face, inference providers | hugging face, face inference |
| apple, iphone, airpods | **gpu, kernels, cuda** |
| stocks, chinese, hong kong | **transformers, ocr, transformers js** |
| battery, batteries, nuclear, ev | **trl, batching, continuous batching** |
| trump, canada, tariffs | **lerobot v0, agents** |
| blockchain, sec | **embedding, sentence, supervised** |
| climate, planes, warming | **leaderboard, evaluation, asr** |
| headphones, garmin, trackers | trump, canada, tariffs |

**fastembed resolves finer technical distinctions** — CUDA kernels,
continuous batching, ASR evaluation, late-interaction embeddings are all
separate topics for it, where model2vec keeps them inside broader
"ai"/"coding" clusters. For a technology-news product that grain is worth
something.

Both keep a legitimate "google, spirit airlines" cluster. That is a real
story (Google bought Spirit's data at a bankruptcy auction), not the
word-attractor failure TF-IDF had — the earlier problem was *EmbeddingGemma*
landing in it, and neither dense backend does that.

**Coverage is the honest weak point: only 25–30% of articles fall into any
kept cluster.** HDBSCAN leaves 670–873 as noise and the coherence prune
removes about half the rest. A taxonomy that classifies a quarter of the
corpus is not yet a replacement for the 13 LLM categories; it is a
high-precision signal over part of the feed.

## Absorbing unseen articles (build on older 60%, classify newer 40%)

| | model2vec | fastembed |
|---|---|---|
| raw clusters → kept | 40 → 20 | 40 → 20 |
| build-set coverage | 28% | 23% |
| absorbed at sim ≥ 0.30 | 40.9% | **58.8%** |
| absorbed at sim ≥ 0.40, margin ≥ 0.02 | 20.9% | **33.0%** |
| median best-match | 0.265 | **0.336** |
| vs a random cluster | −0.010 | −0.003 |

fastembed absorbs unseen articles substantially better — 58.8% against
40.9% at the loose threshold, and a higher median match throughout. Both
score ~0 against a random cluster, so the matches are real.

## Hot news via fine clusters

Second pass inside each rough cluster, single-linkage at cosine ≥ 0.75 —
a rough cluster is a *topic*, a fine cluster should be one *event*.

This works, and the standout is the same story for both backends:

| story | model2vec | fastembed |
|---|---|---|
| **Unitree humanoid-robot IPO** | 11 articles, 3 sources | **15 articles, 4 sources** |
| Strategy / Bitcoin purchases | 4 articles, 1 source | 5 articles, 1 source |
| Hugging Face inference providers | 2 articles, 1 source | 5 articles, 1 source |
| Apple camera AirPods leak | **5 articles, 4 sources** | not surfaced |
| Claude text watermarking | not surfaced | **3 articles, 3 sources** |
| ChatGPT for Teens | 4 articles, 3 sources | — |

The Unitree IPO is exactly the intended signal: 15 articles across gnews,
BBC, Guardian and Nikkei, all covering one event, on a day it was genuinely
the biggest story in the feed. Cluster size plus **source count** is the
usable metric — size alone rewards a single outlet publishing a series, as
with the four `openai_blog` "ChatGPT Work" posts, which are not hot news.

fastembed found the larger cross-source groups; model2vec found more groups
overall (13 vs 7 on the holdout build) but skewed single-source.

## Where this leaves the two candidates

| | model2vec | fastembed |
|---|---|---|
| resident memory | **91 MB** | 172 MB |
| per article | **0.2 ms** | 96–300 ms |
| category grain | broad | **finer, more technical** |
| absorption of unseen | 40.9% | **58.8%** |
| cross-source hot news | weaker | **better** |

Both fit the VM. fastembed is better at the job on every quality axis
measured; model2vec is roughly 500x cheaper per article and half the
memory. Neither is disqualified, and the choice is a product call about
whether the finer grain justifies the cost.

## Reproducing

```powershell
conda activate myfirstagent
python docs/analysis/tools/build_taxonomy.py --backend model2vec --center --save
python docs/analysis/tools/build_taxonomy.py --backend fastembed --center --save
python docs/analysis/tools/build_taxonomy.py --backend fastembed --center --holdout
```

`--save` writes the centroids, the stored mean, and the coherence cutoff to
`docs/analysis/data/`. That file is what a runtime classifier would load;
it is ~50-100 KB.

---

# Testing the rough cluster as a retrieval bucket, 2026-08-19

The proposed use of the taxonomy is not "label each article". It is
two-stage retrieval: a subscriber asks about AAOI, the interest maps to a
rough cluster (say telecom/optical), every article in that cluster is
pulled, and a fine step or an LLM narrows what comes back. Under that
design the taxonomy never needs an AAOI cluster.

That changes what has to be measured. Not classification accuracy —
**coarse recall**: of the articles genuinely relevant to an interest, what
fraction is inside the bucket we pulled? A later fine step can fix
precision; nothing downstream can recover an article the coarse pull never
returned.

Measured with `docs/analysis/tools/measure_routing.py`, model2vec, against
keyword ground truth for real subscriber interests taken from the live DB.

## Bucket routing does not work

| interest | relevant | bucket recall | in **no** bucket at all |
|---|---|---|---|
| Bitcoin | 70 | **81%** | 17% |
| 光通訊 | 7 | 14% | 86% |
| robotics | 104 | 12% | 62% |
| semiconductors | 38 | 8% | 79% |
| AI | 726 | 3% | 61% |
| quantum computing | 36 | **0%** | 97% |
| AAOI | 5 | 0% | 100% |

Three distinct failures, only one of which is fixable by tuning:

1. **The coverage ceiling.** Only 28% of articles are in any kept bucket —
   HDBSCAN discards the rest as noise and the coherence prune halves what
   survives. 61–100% of every interest's relevant articles are therefore
   in no bucket at all and are unreachable by *any* routing. This alone
   caps the design.
2. **Relevant articles spread across buckets.** "robotics" routes to
   `[robotics, autonomous, world robot]` (13 articles) while the larger
   robotics bucket is `[unitree, robot, humanoid]` (24). Top-1 routing
   picks one of several correct answers. Fixable with top-K.
3. **Short interest strings route to noise.** "quantum computing" routes
   to `[comcast, motion, home, fi]` at similarity 0.134 — an argmax over
   values that are all effectively zero. A 4-character ticker or a
   two-word topic has too little text to match a centroid built from
   paragraphs.

## Direct kNN over articles beats it, without any clusters

Same interests, same embeddings, no taxonomy — just rank every article by
similarity to the interest string:

| interest | bucket recall | kNN top-50 | kNN top-200 |
|---|---|---|---|
| quantum computing | 0% | **94%** (68% prec) | 100% |
| robotics | 12% | 47% (98% prec) | **90%** |
| Bitcoin | 81% | 67% (94% prec) | **97%** |
| semiconductors | 8% | 55% | **74%** |
| 光通訊 | 14% | 57% | 71% |

Quantum computing is the clearest case: the bucket returns nothing, direct
search returns 94% at 68% precision. The information was in the embeddings
the whole time — the clustering step threw it away.

**Conclusion: clusters are a good hot-news detector and a poor retrieval
index.** They earn their place on the corroboration signal measured above
(the Unitree IPO at 15 articles across 4 sources), not in the retrieval
path. Retrieval should query article vectors directly.

## Where BM25 belongs: it is not a tie-breaker, it is a different failure mode

Top-50 recall, same ground truth:

| interest | BM25 | embeddings | RRF hybrid |
|---|---|---|---|
| quantum computing | **100%** | 94% | **100%** |
| Bitcoin | **71%** | 67% | **71%** |
| robotics | 40% | 47% | **48%** |
| semiconductors | 13% | **55%** | 45% |
| 光通訊 | **0%** | **57%** | 57% |

The two are complementary in a way that is specific and predictable:

- **BM25 wins when the interest is the literal word in the article** —
  "quantum" appears in quantum articles, so lexical match is perfect and
  embeddings can only approximate it.
- **Embeddings win when it isn't** — semiconductor articles say "chip",
  "foundry", "TSMC", "wafer", not "semiconductors", and BM25 collapses to
  13% while embeddings hold 55%.
- **The 光通訊 row is the one that matters most for this product.** BM25
  scores exactly 0 because the interest is Chinese and the corpus is
  English — no shared token exists, so lexical retrieval is *structurally*
  incapable, not merely weak. Embeddings get 57%. This project has
  subscribers whose stored interests are `機器人科技`, `科技財經`, `光通訊`,
  so cross-language retrieval is a live requirement, not a hypothetical.

RRF hybrid takes the better of the two in four rows of five and loses ten
points to embeddings alone on semiconductors — a reasonable trade for not
having to know in advance which mode an interest needs.

## AAOI: a retrieval result that is really an ingestion finding

AAOI scores 0% by every method, including BM25, which should be its best
case — a ticker symbol is exactly what lexical search is for. The reason
is that **the corpus contains zero articles mentioning AAOI or Applied
Optoelectronics**. The 5 "relevant" articles matched only on the broader
`optical`/`transceiver` keywords.

No retrieval method can return what was never fetched. Related counts in
the same 2,262-article snapshot:

| interest | articles in corpus |
|---|---|
| robotics | 100 |
| Bitcoin | 60 |
| quantum computing | 36 |
| semiconductors | 13 |
| 光通訊 / optical | 6 |
| AAOI / AOI | **0** |
| Edge AI boards | 1 |

The sources that query by subscriber interest are the ones meant to cover
exactly these low-profile topics, and they are the ones currently
degraded: Perigon is 403ing on an exhausted monthly quota (see
`docs/plans/security-plan.md` finding 21) and NewsAPI runs once a day with
a 24–36 h free-tier delay. **The niche-interest gap is an ingestion
problem wearing a retrieval problem's clothes**, and no amount of
classifier work addresses it.

---

# Does human-merging the categories help? 2026-08-19

First a distinction that matters for the question: **kNN and BM25 do not
produce categories.** They are retrieval — given a query, they rank
articles. Categories come from the clustering step. But there is a real
design where a human writes the category list and retrieval fills it, so
both were measured:

- **A — discover then merge**: cluster, then a human merges fragments into
  real categories (`trump/iran` + `trump/tariffs` + `abc/disney/fcc` →
  Politics).
- **B — name then retrieve**: a human writes the category names first and
  each category's members are whatever kNN returns for its name. No
  clustering in the path.

Measured with `docs/analysis/tools/measure_merged_categories.py`, merging the
36 model2vec clusters into 8 categories.

## Merging fixes exactly one of the three failure modes

| interest | relevant | unmerged cluster | **A: merged** | B: named category | direct kNN on the interest |
|---|---|---|---|---|---|
| robotics | 104 | 12% | **36%** | 90% | **90%** |
| Bitcoin | 70 | 81% | **83%** | 91% | **97%** |
| semiconductors | 38 | 8% | 8% | 24% | **74%** |
| quantum computing | 36 | 0% | 0% | 11% | **100%** |
| 光通訊 | 7 | 14% | **0%** | 14% | **71%** |

Merging does what it was predicted to do and nothing more:

- **Fragmentation: fixed.** Robotics tripled, 12% → 36%, because the two
  sibling clusters (`robotics, autonomous` and `unitree, robot, humanoid`)
  became one bucket. This is real and it is the whole benefit.
- **The coverage ceiling: unchanged.** 629/2262 articles are in a merged
  category — 28%, the same 28% as before. Merging combines buckets; it
  adds no articles to them. The 72% HDBSCAN discarded as noise are still
  unreachable.
- **Short-query routing: made worse.** 光通訊 went 14% → **0%**, routed to
  "Energy". A merged centroid is the average of a broad mixture, which is
  *harder* for a short string to match than a specific cluster centroid,
  not easier. Quantum computing routes to "Health". Merging widens the
  buckets and blurs exactly the signal a two-word query needs.

## Any category layer between the interest and the articles costs recall

The last two columns are the finding. Design B (name the category, then
retrieve for that name) beats merged clusters everywhere — but querying
with the **subscriber's own interest string** beats B everywhere too, and
by a lot: 100% vs 11% on quantum computing, 74% vs 24% on semiconductors,
71% vs 14% on 光通訊.

The reason is structural. The interest *is* the query. Routing it through
a category first replaces a specific query with a general one, and
whatever specificity distinguished "quantum computing" from "semiconductors"
is discarded at that step. No later stage can recover it.

## So what are categories still for?

Not retrieval. Three things the measurements do support:

1. **Browsing.** A subscriber who doesn't know what to ask needs a list to
   pick from. That is a display need, and 28% coverage is acceptable for
   it in a way it is not for retrieval.
2. **Hot-news detection.** Fine clusters inside a rough cluster found the
   Unitree IPO at 15 articles across 4 sources. That signal needs the
   grouping.
3. **Digest diversity.** Spreading a push across categories, once they
   exist.

One caveat even for browsing: the merged categories are badly lopsided —
**AI & ML holds 364 of 629 articles (58%)** — because the corpus is (see
the source-composition finding above: 47% of articles come from
AI-only or AI-skewed feeds). A human merge inherits that skew; it does not
correct it.

## What this implies for the month-long collection

More data raises the ceiling — HDBSCAN's `min_cluster_size=8` is why
semiconductors (13 articles) and optical (6) can't form clusters today,
and 30× the corpus fixes that arithmetic. It does **not** change the
finding that a category layer is lossy for retrieval, since that is
structural rather than a sample-size artifact. Expect the month of data to
improve browsing categories and hot-news detection, and to leave
"retrieve with the interest string directly" as the right retrieval path.

---

# Correction: the 28% coverage was an artifact, and it invalidates two conclusions above

Everything above measured stage 1 by using **HDBSCAN's own labels as the
article-to-category assignment**. That was wrong. HDBSCAN marking 72% of
points as noise is a statement about *density* — it declines to say those
points form clusters. It is not a statement about which centroid an
article is nearest to. The centroids exist; every article can be assigned
to its nearest one.

The design being built is a funnel, and stage 1's job is completeness, not
precision:

    stage 1  rough categories -- a complete partition, like a newspaper's
             section list. Every article gets one or several. "Other" is
             allowed but must stay small.
    stage 2  filter the narrowed set by the subscriber's interest, cheaply
    stage 3  hot spots by cluster size; novelty by distance from what's
             already selected

Re-measured with nearest-centroid assignment
(`docs/analysis/tools/measure_full_partition.py`):

| assignment policy | coverage | "Other" |
|---|---|---|
| HDBSCAN labels (what was measured above) | 27.8% | 72.2% |
| nearest centroid, sim ≥ 0.05 | **98.1%** | **1.9%** |
| nearest centroid, sim ≥ 0.10 | 88.3% | 11.7% |

## Two conclusions above are withdrawn

**"Human merging inherits the corpus skew — AI & ML holds 58%."** Also an
artifact. Under a real partition the categories are close to balanced:

| category | share | | category | share |
|---|---|---|---|---|
| AI & ML | 22.1% | | Entertainment & Media | 8.5% |
| Crypto | 10.5% | | Energy & Climate | 7.5% |
| Consumer devices | 9.8% | | Politics & Policy | 7.3% |
| Robotics | 9.8% | | Security | 6.9% |
| Markets & Stocks | 9.3% | | Health & Science | 6.5% |
| | | | **Other** | **1.9%** |

22% AI in a technology news feed is unremarkable. This is a usable section
list.

**"Any category layer between the interest and the articles costs recall."**
Overstated. Stage-1 recall with a complete partition:

| interest | HDBSCAN labels | nearest centroid | multi-label | routed to |
|---|---|---|---|---|
| Bitcoin | 83% | 99% | **100%** | Crypto |
| robotics | 36% | 86% | **89%** | Robotics |
| AI | 28% | 48% | **58%** | AI & ML |
| quantum computing | 0% | 28% | **47%** | Health & Science |
| semiconductors | 8% | 32% | **45%** | Markets & Stocks |
| 光通訊 | 14% | 14% | 14% | Entertainment & Media |

Bitcoin at 100% and robotics at 89% are fine for a first-stage filter.

**Multi-label assignment earns its place**: every interest improves by
3–19 points, at 1.61 categories per article. One story genuinely is AI and
Software and Industry at once, and forcing a single label discards that.

## What is actually still broken, and why it waits for more data

The remaining failures share one cause: **there is no category for the
topic**. 光通訊 routes to Entertainment & Media and semiconductors to
Markets & Stocks because the taxonomy has no Telecom/Networking and no
Hardware/Semiconductor category — and it has none because the corpus holds
6 optical articles and 13 semiconductor ones, under HDBSCAN's
`min_cluster_size` of 8.

That is a corpus-size problem, not a method problem, and it is what the
month of archived collection is for. Deferred until then.

(The merge rules here also correct an earlier error: an ABC/Disney/FCC
story is the entertainment industry, not politics. A regulator appearing
in a story doesn't make the story political when the subject is who owns
a broadcaster.)

---

# The cost premise changed, 2026-08-21

Everything above was motivated by removing the per-article API cost of
LLM classification. That premise has been measured and does not hold at
this volume.

The live system classifies **991 articles/day in 20 LLM calls**, costing
roughly **$0.03/day — under $1/month**. Full figures and the cheaper
levers (prompt caching; the unused `LLM_MODEL_CLASSIFIER` env var) are in
`docs/plans/taxonomy-and-admin-plan.md`.

So the measurements here should be read as answering a **quality**
question, not a cost one:

- a taxonomy derived from the corpus rather than hand-written
- hot-news detection by cluster size and source count (the Unitree IPO at
  15 articles across 4 sources)
- the far-from-everything diversity pick

Each of those was measured to work. None of them is worth building to
save a dollar a month, and this note exists so a future reader doesn't
re-derive the cost argument from the enthusiasm in the sections above.

---

# Hot topics and novelty for push selection, 2026-08-21

Embeddings here are **not for classification** — the LLM does that and the
taxonomy lives in the database. This is about which of the articles that
survive filtering are worth sending. Two signals:

- **hot** — several outlets are paying attention to the same thing
- **novel** — one or two articles unlike everything else selected, so a
  digest isn't five versions of one story

Measured with `docs/analysis/tools/measure_hotspots.py` on a 2,706-article
snapshot, 1,127 with real categories — the first snapshot taken since
classification was repaired, so the first where a filtered set is big
enough to say anything.

## "Hot" is two different things, and one mechanism cannot find both

### Story-level: the same event, several outlets

Semantic clustering at a high threshold finds this. Inside the AI
category (531 articles), cosine ≥ 0.75 gives 8 groups.

**Size alone is actively misleading.** The largest groups are all one
outlet publishing a series:

| category | n | largest group | sources |
|---|---|---|---|
| Hardware | 196 | **30 articles** | **1** (newsapi, a Chinese finance series) |
| Robotics | 138 | 11 articles | 1 (gnews) |
| AI | 531 | 2 articles | 2 (Google buys Spirit Airlines' data) |
| Finance | 252 | 2 articles | 2 (Travelodge CEO resigns) |
| Policy | 156 | 2 articles | 2 (Evergrande founder sentenced) |

A 30-article group from one source outranks every genuine multi-source
story. **The usable metric is distinct source count, not article count** —
and with that filter applied, real corroboration at this volume is
currently pairs, not the 15-article/4-source Unitree IPO seen in an
earlier snapshot.

### Topic-level: several articles about the same subject, saying different things

This is the case that matters more day to day — "GPT-5.3 shipped and five
articles reference it while discussing different things", or "AI security
is getting a lot of attention this week". **Semantic clustering cannot
find it, and lowering the threshold does not help.**

Two measurements say so.

Articles sharing an entity are not semantically similar:

| term | AI articles mentioning it | mean pairwise cosine | pairs ≥ 0.75 |
|---|---|---|---|
| openai | 16 | +0.190 | **0** |
| chatgpt | 12 | +0.256 | **0** |
| anthropic | 6 | +0.311 | **0** |

Not one pair clears the story threshold. They are about different things
and merely share a token.

And lowering the threshold produces chaining collapse, not topics:

| threshold | groups | largest group | share of the category |
|---|---|---|---|
| 0.75 | 8 | 26 | 5% |
| 0.60 | 19 | 28 | 5% |
| 0.50 | 26 | 42 | 8% |
| 0.40 | 30 | **180** | **34%** |
| 0.30 | 5 | **452** | **85%** |

Single-linkage goes from fragments straight to a blob with nothing usable
in between. There is no "topic threshold" to find.

**So topic-level hotness is a lexical/entity problem, not a semantic
one** — which is the honest answer to why BM25 was never redundant with
embeddings here. Each finds something the other structurally cannot.

## Burst detection needs history the active cache does not have

The standard approach is comparing a term's rate in a recent window
against its rate in a baseline period; [Kleinberg's burst
detection](https://www.cs.cornell.edu/home/kleinber/bhs.pdf) formalises
this as a state machine over the stream, and simpler rate-ratio versions
are the usual practical starting point.

A first attempt failed for a structural reason worth recording: the active
cache has a 48-hour TTL and one ingestion cycle can add 1,000+ articles, so
"the last 24 hours" captured 1,010 of 1,127 articles and the baseline was
117. Every candidate term had a historical count of zero, making the lift
ratio meaningless, and the top results were stopwords ("after", "can",
"now").

**The archive is the baseline.** `NEWS_ARCHIVE_DIR` has been accumulating
since 2026-08-19 precisely so there is history to compare against; this is
the first thing that actually needs it. Burst detection should read the
archive for the baseline and the live cache for the recent window, not
try to split the live cache against itself.

## Time decay: what to weight and how

Hotness is inherently time-sensitive — a story with five sources yesterday
should outrank one with five sources last week — so the weight of evidence
has to fade. Three families, all long-established:

### 1. Rank-level decay (the Hacker News / Reddit family)

    score = evidence / (age_hours + 2) ^ gravity        gravity ≈ 1.8

[Hacker News](https://www.righto.com/2013/11/how-hacker-news-ranking-really-works.html)
uses this shape. Time is raised to a higher power than the evidence, so
nothing stays hot indefinitely regardless of how much evidence it
accumulated. Cheap, stateless, one tunable.

**Best fit for the story-level signal here**: a cluster's score becomes
`distinct_sources / (hours_since_newest + 2) ^ gravity`, computed on the
fly from data already in hand. Nothing to persist.

### 2. Exponential ageing of the evidence itself (the damped-window family)

    w(t) = 2 ^ (-λ · Δt)

Each observation carries a weight that halves every `1/λ`. This is what
stream-clustering algorithms use to let clusters fade —
[DenStream](https://www.cs.sfu.ca/~ester/papers/SDM2006.DenStream.final.pdf)
maintains micro-clusters under exactly this damped window, and CluStream
keeps time-horizon snapshots for the same purpose.

**Best fit for the term-burst signal**: instead of a hard 24h/baseline
split, every article contributes `2^(-λ·age)` to each of its terms. A term
mentioned by three articles today outweighs one mentioned by three
articles four days ago, with no window boundary to tune and no cliff at
the edge of it. It also fixes the failed measurement above, which was
entirely an artefact of where the window boundary fell.

### 3. Stateful stream clustering (DenStream/CluStream proper)

Maintain micro-clusters incrementally with decaying weights, promoting and
retiring them as the stream moves. This is the fully general answer and it
is the wrong tool here: it exists for streams too large to re-cluster, and
this corpus is ~1,000 articles a day that re-cluster in 0.2 s with
model2vec. The cost of maintaining incremental state would exceed the cost
of recomputing.

### Recommendation

Use (1) for story clusters and (2) for term bursts, and skip (3). Both are
arithmetic over data already being collected — no new storage, no
incremental state, nothing to keep consistent across restarts.

Two parameters to fit from data rather than pick: `gravity` for the rank
decay, and the half-life for term weighting. A sensible starting point is
a half-life near the push interval, so a story stays hot for roughly one
digest and then fades, but that is a guess and should be replaced by
looking at how long real stories actually stay covered in the archive.

## Novelty works, but "farthest" is not the same as "interesting"

Distances are discriminating, which is the thing TF-IDF could not do (its
median max-similarity across a candidate pool was 0.0000, with 61% ties):

| category | pool | median max-sim to a 5-article selection | exact zeros |
|---|---|---|---|
| AI | 526 | 0.138 | 37 |
| Research | 424 | 0.168 | 5 |
| Finance | 247 | 0.156 | 2 |

Farthest scores −0.083 against a nearest of 0.415 — a real ordering, not a
tie-break.

**But the articles it selects are outliers for the wrong reasons.** The
farthest article in AI was a truncated headline ("plus infectious disease
experts to Gujarat, with AI"); in Finance it was a Chinese-language fund
prospectus in an English-dominant pool. Mis-parsed titles and
language outliers are maximally distant from everything by construction,
so a naive farthest-point pick will surface junk before it surfaces
genuine novelty — and it would do so on every single digest.

Novelty selection needs a quality floor first: a parseable title of
reasonable length, and a language consistent with the subscriber's
digest. Only then take the farthest survivor.

## Duplicates: the fix is not title matching

47 articles share 17 duplicate titles, and **15 of the 17 are one source
repeated** — but they are two different problems:

- **Genuine duplicates**: 9 gnews copies of one syndicated wire story
  under different URLs. Link-based dedup cannot see these.
- **Not duplicates at all**: "Tech Now", "Tech Life", "Business Daily" —
  BBC programme titles, where each episode is different content under a
  recurring name.

Deduplicating by title would wrongly collapse the second group. The
distinguishing signal is content similarity, which the embeddings already
provide: collapse near-identical vectors (cosine above ~0.95) to one
representative at candidate-selection time. That serves both purposes at
once — subscribers stop receiving the same wire story twice, and the
story-level hotness signal stops counting one syndicated piece as nine.

---

# Correction, 2026-08-24: both "junk" exhibits were measurement artifacts

The section above concluded that "a naive farthest-point pick will surface
junk before it surfaces genuine novelty -- and it would do so on every
single digest," and required a quality floor before novelty selection could
be trusted. It rested on two exhibits. Neither survives inspection: both
are properties of the measurement harness, not of the corpus.

This is the third finding in this project with that shape, after
`service.name=unknown_service` (a dead man's switch watching a service name
production never set) and the 28% coverage artifact corrected earlier in
this same document. The common cause each time: **the harness did not
reproduce what production actually does.**

## Exhibit 1 -- "plus infectious disease experts to Gujarat, with AI"

Not a mis-parsed headline. The snapshot collector cut it.

`fetch_cache_snapshot.py` read each field with `grep -m1 '^title:'`, which
takes only the first *physical* line. PyYAML wraps at 80 columns by
default, so every title longer than that arrived pre-truncated. Reproduced
exactly, with the real headline:

```
title: CIDSCON 2026 to bring 950-plus infectious disease experts to Gujarat, with
  AI and AMR in focus
```

`grep -m1` yields `CIDSCON 2026 to bring 950-plus infectious disease experts
to Gujarat, with`.

The corpus-wide symptom was there to be seen and was missed: **the longest
title in a 2,706-article snapshot is 94 characters, and every one of the 21
sources has a p95 near 80** -- including arxiv, whose paper titles routinely
run past 150.

The full headline is a genuinely good pick for an AI subscriber. A medical
conference where AI and antimicrobial resistance are on the agenda is
exactly the "in the area, unlike a typical article in it" shape novelty
selection is supposed to find. **The vector was right; the string was
broken.**

Fixed: `yaml_get()` in that collector now joins the continuation lines, with
a `command -v awk` guard so a container without awk fails loudly instead of
emitting empty fields.

## Exhibit 2 -- the Chinese-language fund prospectus in an English pool

Real, but production already excludes it. 65 of the 66 non-Latin-script
titles in the snapshot come from `newsapi`, which is in
`news_sources.RESTRICTED_SOURCES`; `news_push.select_candidate_articles`
skips restricted sources unless `include_restricted` is set, so **not one of
those articles can reach a digest today.**

`measure_hotspots.py` has no source filter at all -- no mention of
`RESTRICTED_SOURCES`, `newsapi` or `perigon` anywhere in it. It ranked
novelty over a pool containing 65 articles production would never send.

## What this invalidates

The quality floor the section above called for was already in production for
the case it was written about. The remaining gap was never in the selection
logic; it was that the harness measured a different pool than the bot
serves.

The narrower true finding survives: a language outlier *in the vector space*
is maximally distant from an English-dominant corpus by construction. That
matters for embeddings, clustering and centroids -- which those articles
were still polluting, because being excluded from a digest is not the same
as being excluded from the cache.

## Decision taken, 2026-08-24: drop non-Latin scripts at ingestion

`news_ingest.is_latin_script` now gates every article before it is cached,
so a non-English article is never embedded, clustered or classified. It
drops 66 of 2,706 titles (2.4%) on the snapshot, 65 of them from `newsapi`.

Deliberately a **script** test, not a language test. It drops Chinese,
Japanese, Korean, Arabic, Cyrillic, Devanagari, Thai and Hebrew. It does not
drop Spanish, French or German, and is not meant to -- that leakage is
accepted rather than chased:

- a language-detection library is a new dependency for a job a character
  scan does, and this repo has been bitten once already by a dependency's
  transitive imports (CLAUDE.md, on `arize-phoenix`);
- the zero-dependency substitute, scoring English function words, was
  measured against this snapshot and **misfired on 7% of titles**, because
  headlines drop function words and arxiv titles barely use them at all
  ("Fast high-dimensional mean testing via logistic regression" reads as
  non-English to it).

Gating at ingestion rather than at selection is the reversibility trade
being made knowingly: a filtered-at-selection article is still available to
a later change of mind, a never-cached one is not. The call was that a clean
corpus is worth more than that option.

## The corrected offbeat score: a gate, not a sum

"Offbeat" is not the same as "novel", and neither is a genre label. The
distinction that drives the formula: *AI + leak* is ordinary AI news, while
*AI + stray cats* is offbeat. Offbeat is an unexpected **pairing**, and
pairings cannot be enumerated in advance the way genres can -- which is
precisely why it has to be a distance rather than a tag.

The obvious two-term score is wrong:

    score = sim(article, interest_query) - sim(article, area_centroid)

A linear combination lets a high query-match buy back a high centroid-match.
Measured on the cleaned pool, the top AI result under this score was
`'Agent Applications: A Reference Architecture for AI Agent Systems'` at
centroid similarity **+0.622** -- about as central as an article can be.

The working form is a **gate followed by a ranking**:

1. keep articles whose similarity to the interest query is at or above the
   area median (they must still be in the area);
2. among the survivors, rank by *lowest* similarity to the area centroid.

Measured with model2vec `potion-base-8M` on the cleaned pool (838 articles:
2,706 less 65 restricted, 349 truncation-suspect, 1 non-Latin, 1,437 with no
category, and exact-duplicate titles):

| area | pool | survive gate | top picks by lowest centroid similarity |
|---|---|---|---|
| AI | 393 | 197 | `AI may force govt to renegotiate bilateral tax treaties` (c=+0.267); `Run frontier LLMs on sovereign EU infrastructure` (+0.170); `'Mjolnir: Automated Cross-Vendor Adversarial Review'` (+0.257) |
| Hardware | 126 | 63 | `India, Singapore sign pacts on telecom, food security; discuss SMRs, chips` (+0.282); `Search for Majorana Bound States in Short Chains of Proximitised Quantum...` (+0.270) |
| Finance | 180 | 90 | `Moderna and Merck shares soar on mRNA cancer vaccine` (+0.206); `'FinRCA-Bench: Benchmarking Evidence Retrieval and Reasoning for Financial...'` (+0.224) |

For contrast, the most central AI articles score c=+0.731 (`AI Influence
Level (Ail) v1.0`) and +0.707 (`The AI Debate Is About Control, Not AI`).

*AI x international tax law* and *chips x diplomacy x small modular reactors*
are the target shape. Honest assessment: **roughly three of five picks per
area are genuinely offbeat, and none are garbage.** The weaker survivors
(`Seel, Which Manages $6 Billion Of GMV Annually, Launches Resale` in AI,
`The Pixel 11 paradox` in Hardware) are there because their category tags
are wrong -- a classification problem, not a scoring one.

## Still open

- **Re-pull a snapshot with the fixed collector.** Every number in this
  document above the correction line was computed on truncated titles. The
  349 articles held out as truncation-suspect here are mostly not truncated
  at all; they were simply longer than 78 characters.
- **Whether any real source-side truncation exists** is unknown, and cannot
  be known until that re-pull. Do not build a repair for it first. Of the
  zero-token detectors tried, only three are worth keeping: a trailing
  ellipsis, a trailing comma, and merging title with summary on their
  longest overlap (which recovers the Gujarat headline, but fired on only 4
  rows corpus-wide, 3 of them duplicates of each other). The "headline opens
  with a continuation word" rule flagged 14.7% of the corpus and is unusable
  -- it catches every title starting with "A" or "The".
- **Exact-duplicate articles are in the cache**, not just near-duplicates:
  rows 1191 and 1224 of the 0821 snapshot are the same article. The cosine
  >= 0.95 collapse this document argues for is still wanted, and would
  subsume this.
- **The gate threshold is the area median**, chosen for having no better
  reason. It should be fitted once there is a clean corpus to fit it on.

## Shipped, 2026-08-25: embeddings landed in production

The near-duplicate collapse and the offbeat gate/rank score described
above are now implemented and wired into the live pipeline, not just
measured on a snapshot. `news_embed.py` (model2vec `potion-base-8M`,
per this document's own backend comparison), `news_cache.write_article`
storing one embedding per article, `news_ingest.py` computing it at
ingestion time, and `news_push.select_candidate_articles`/
`_pick_for_topic` applying `NEAR_DUPLICATE_SIMILARITY` and
`OFFBEAT_SLOTS_PER_TOPIC` per subscriber-topic.

Two adaptations from what was measured here, both load-bearing:

- **"Area" is now a per-topic candidate pool, not one of the 13 named
  categories.** Push moved to one message per subscriber interest
  (2026-08-24, `docs/plans/bot-features-plan.md`) before this landed, so
  there is no fixed "AI"/"Finance"/"Hardware" pool to compute a centroid
  over any more -- the centroid and gate are computed fresh, per topic,
  per push, over that topic's own candidate pool
  (`OFFBEAT_POOL_SIZE = 30`).
- **The gate is the REMAINDER's own median**, not the pool's -- the
  remainder (what's left after the recency cut) is what the offbeat
  slots actually choose from, so that's what needs a meaningful split.

**Still not done, unchanged from above**: the gate threshold remains a
provisional choice, not a fitted one -- shipping it doesn't fit it. A
fresh snapshot pull with the fixed collector, and fitting this threshold
against it, are both still open. `hot` (story-level clustering across
outlets) was explicitly out of scope for this pass; only near-duplicate
collapse and offbeat/novelty selection shipped.

## Stage-1 retrieval precision: the fine filter, 2026-08-25

A live bug exposed the gap this section closes: a subscriber with four
interests (`AI`, `AI Agent`, `AI coding`, `Large Language Model`) got four
push messages, three of them titled with the same generic
`AI產業趨勢報告` fallback and carrying near-identical article lists. Root
cause was structural, not cosmetic: all four interests map to the same
coarse category (`AI`), and nothing after that coarse filter narrowed the
pool per interest -- `write_push_digest` received the same ~200-article
candidate set for all four topics and had no signal to tell them apart.
This is the same finding as [*Any category layer between the interest and
the articles costs recall*](#any-category-layer-between-the-interest-and-the-articles-costs-recall)
from the other direction: the coarse category is necessary for recall, but
recall alone isn't precision, and nothing downstream was providing the
missing precision layer.

**The `[topic]` prefix made this worse, not better.** `write_push_digest`
prefixed every candidate line with `[AI Agent]`/`[AI coding]`/etc., visually
presenting the *coarse* category tag as if it were confirmed per-article
metadata. Reproduced live: the same candidate list under `topic="AI Agent"`
vs. `topic="AI coding"` produced near-identical digests with the model doing
essentially no filtering of its own -- the prefix read as "already sorted,"
so the model trusted it instead of judging relevance itself. Removed; the
topic is now stated once in the system prompt with explicit
skepticism-calibration language instead of repeated per-line as pseudo-metadata.

### A fixed similarity threshold cannot work across topics

First attempt was a fixed cosine cutoff after the coarse filter. Measured
against real per-topic queries on the same corpus: "AI Agent" genuine
matches scored 0.53-0.72 cosine; "Large Language Model" genuine matches
topped out at 0.166 on the *same* corpus. Absolute cosine scores are not
comparable across different queries with model2vec -- there is no single
threshold that works for both. This ruled out any fixed-threshold design
and motivated a relative cut instead (a percentile/gate on the pool's own
score distribution), which became the shape everything below refines.

### Recall-based measurement, not raw keep-counts

Early self-checks eyeballed keep-counts and top-ranked titles. Corrected
methodology, stated directly: *if a topic genuinely has only one relevant
article in the pool and everything else gets filtered out, that is the
correct result -- filtering hard is not itself a problem. The actual test is
recall: if 5 articles in the pool are genuinely about "AI coding" and 8 are
genuinely about "AI Agent," the filter must keep all 5 and all 8. Keeping
10 or 16 (i.e. some false positives alongside them) is fine. Losing even
one of the 5 or 8 is not.*

Building a real ground-truth set to measure this against caught two of my
own labeling errors, both corrected directly by spot-checking summaries
rather than trusting titles:

- **Over-inclusion by loose association**: articles about a Cowork "memory
  feature" were first counted as "AI coding" ground truth because Cowork is
  a coding-adjacent tool -- wrong; the articles are about a generic
  memory/privacy feature, not about coding. Correction as stated: *not*
  "Cowork is unrelated," but "Cowork is not *strongly* related" -- relevance
  is a matter of degree, not a binary in/out flag, and the ground truth set
  needs to reflect that.
- **Under-inclusion by headline style**: two low-ranked "AI coding" ground
  truth items (Virgin Atlantic, Asana) were initially waved off as
  business-outcome headlines nobody reading "AI coding" would care about.
  Pulling their actual summaries showed they genuinely describe concrete
  coding/testing work ("near-total unit test coverage," "replace an
  outdated testing system") -- legitimately on-topic, just written in a
  business-outcome headline style with no technical vocabulary in the
  title. This is a real, distinct failure mode from the Cowork case: not
  "loosely related," but "correctly related, harder for a title-only
  embedding to see."

That second finding is what made title+summary embedding non-negotiable
rather than a nice-to-have (see below): a title-only vector has no way to
see past a headline-writing-style gap when the actual on-topic content
lives in the summary.

### What model2vec (static embeddings) cannot do

Two capability ceilings, confirmed empirically rather than assumed:

- **No negation.** Appending "not about memory/privacy/pricing/UI" to a
  topic's definition query was tried as a way to push the Cowork articles
  down. Measured effect: their rank got *slightly worse* (closer to the
  top), not better -- the literal word "memory" added to the query vector
  pulled it *closer* to memory-related articles, the opposite of the
  intended effect. Static embeddings have no mechanism to represent "not,"
  so negated instructions in the query text actively backfire. Not
  attempted again.
- **HyDE has fast diminishing returns.** Tested embedding a full
  hypothetical-document outline for a topic against a short one-sentence
  definition. Needed-keep-fraction (the recall metric above) moved from
  44% to 41% -- most of the gain comes from having *any* real definition
  instead of the bare topic phrase, not from how elaborate that definition
  is. `expand_interest_for_retrieval` (`news_classify.py`) generates a
  short glossary-style definition per interest, cached in
  `interest_query_expansions` (`users_db.py`) -- the cheap, mostly-there
  version of this, not the expensive outline.

### Title+summary embedding: a measured trade-off, not a clean win

Confirmed first that no full article body exists anywhere in the system --
`news_sources.py` caps RSS summaries at 300 characters, `hackernews`
articles carry `summary: None`. Title+summary is the richest text
available, not a step toward something richer still available but unused.

Measured switching the embedded text from title-only to title+summary on
the same corpus: Virgin Atlantic's needed-keep rank moved from the 87th
percentile to the 68th, Asana similarly -- both real recall improvements,
matching the headline-style gap found above. But the Cowork memory
article's rank *also* improved, from the 25th percentile to the 12th (i.e.
became *more* prominent, not less) -- the richer text amplifies the
loosely-related false positive along with the genuinely-related items. Net
assessed as worth it (more of the real signal than of the false-positive
signal), but explicitly not a strict improvement on every axis -- the
remaining precision cost is left to `write_push_digest`'s own LLM judgment
as the final layer, per [*Merging fixes exactly one of the three failure
modes*](#merging-fixes-exactly-one-of-the-three-failure-modes)'s established
division of labor between retrieval and generation.

**Shipped**: `news_ingest.py`'s `run_ingestion_cycle` now embeds
title+summary, not title alone, wiring the finding above into the
ingestion path the same day it was measured. No backfill was needed --
the ~48-hour cache TTL (`DEFAULT_TTL_HOURS`) phased out the older
title-only vectors on its own.

### Brand-name matching contributes, but isn't the dominant signal

Ablation: stripped `\bcodex\b` (case-insensitive) from titles and summaries
in the "AI coding" corpus and re-ran the ranking, keeping the un-stripped
text alongside in the output for comparison. Worst-needed-keep rank moved
from the 68th to the 83rd percentile after stripping -- Codex-name matching
is worth roughly 15 percentage points of rank, a real and measurable
contribution, but most mid-rank items barely moved. Brand-name string
matching helps; it is not what the ranking is actually built on.

### The keep-count formula: absolute clamp, not a fixed fraction or a fixed number

Neither a fixed top-N nor a fixed percentage survives contact with real
pool-size variance: a fixed N over-keeps for a small pool and a fixed
percentage over-keeps for a very large one (999 articles in the "AI"
category alone, in a real production-cache pull). Landed on:

```
n_kept = min(RELEVANCE_KEEP_MAX,
             max(round(pool_size * RELEVANCE_KEEP_FRACTION), RELEVANCE_KEEP_MIN))
```

with `RELEVANCE_KEEP_FRACTION = 0.10`, `RELEVANCE_KEEP_MIN = 20`,
`RELEVANCE_KEEP_MAX = 50` -- chosen after reviewing ranked 24-item and
top-50 CSV exports of real production-cache data by hand and confirming top
50 is a reasonable range for what a subscriber-facing digest should draw
from.

The raw pool fed into this formula went through two revisions the same
day. First, `RELEVANCE_SAMPLE_SIZE` (a count-based pre-cap on the
category-matched pool, applied before this formula ever saw it) was
raised from 60 to 600: 60 was never measured, just an unexamined 2x
multiple of `OFFBEAT_POOL_SIZE`, and at that size the 10% fraction can
never exceed `RELEVANCE_KEEP_MAX` for any real topic -- making the MAX
ceiling permanently dead code. Then, once it was clear per-article
embeddings are precomputed at ingestion time (not recomputed per push),
`RELEVANCE_SAMPLE_SIZE` was removed outright rather than re-tuned again:
a recency-based count cap ahead of the relevance filter costs almost
nothing to keep raising, but it also risks discarding a genuinely
more-relevant, slightly-older article before the filter ever scores it,
for a savings (a few hundred more 256-dim dot products) too small to be
worth that risk. The pool this formula now sees is bounded only by
whatever the 48h cache TTL (`DEFAULT_TTL_HOURS`) leaves on disk -- 999
"AI"-category articles were measured live within that window, so the
ceiling still engages for a broad topic, just for a naturally-occurring
reason rather than a tuned one.

`OFFBEAT_POOL_SIZE` was changed from a separately-hardcoded 30 to
`= RELEVANCE_KEEP_MAX` for the same reason applied one level down: a
tighter, silently-binding cap two constants away from where you'd look for
it defeats the outer one without any test noticing.

Two implementation bugs caught only because a real test failed, not by
inspection:

- `int(pool_size * (1 - RELEVANCE_KEEP_FRACTION))` is not the same as
  computing from the keep side: `1 - 0.9 == 0.09999999999999998` in Python
  float arithmetic, so for `pool_size=10` this computed `cut_index=0`
  instead of `1`. Fixed by computing `n_kept` directly and deriving
  `cut_index = len(scored) - n_kept`, never the subtraction-first form.
- Without clamping `n_kept = min(n_kept, len(scored))`, a pool smaller than
  `RELEVANCE_KEEP_MIN` drives `cut_index` negative, and Python silently
  indexes a negative index from the *end* of the sorted-ascending list --
  making the single highest score the gate and excluding almost
  everything, the exact opposite of the intended fail-open behavior for
  small pools.

**Mutation-testing gotcha, worth keeping in mind for any test over
`FakeEmbedder`-scored pools**: `FakeEmbedder` (`tests/fakes.py`) is a
hash-based deterministic word-overlap embedding, and at moderate pool
sizes (~500 items) it produces real tie-clusters wide enough to absorb the
difference between a capped and uncapped keep-count -- a test built at that
scale passed identically whether `RELEVANCE_KEEP_MAX` was applied or
deleted from the code, silently failing to prove the ceiling did anything.
Confirmed by deliberately mutating the formula and rerunning: not caught at
pool=520 (capped and uncapped survivor counts landed in the same tie
cluster), caught cleanly at pool=3000 (208 capped vs. 370 uncapped
survivors, a gap no tie cluster spans). When a test's job is to prove a
numeric ceiling changes behavior, size the fixture from a real diagnostic
run showing separation, not from "big enough to look thorough."

**Still open, same shape as the corpus-side items above**: the gate
threshold inside `_filter_by_relevance` is itself provisional (a relative
cut on the pool's own distribution, described above) -- fitting it against
a large clean corpus, and the `pushed_links`-as-relational-table question
raised alongside this investigation, both remain deferred.

### Real end-to-end confirmation, same day

Ran the actual `news_push._filter_by_relevance` and
`news_classify.expand_interest_for_retrieval` -- real model2vec, real
DeepSeek call, no fakes -- against the full 999-article "AI"-category pool
from the same production-cache pull used throughout this section, for the
three topics that originally collapsed into one generic digest (the bug
that started this investigation): "AI Agent," "AI coding," "Large Language
Model."

Result: each topic now gets a genuinely distinct top-8, on-topic for its
own definition and clearly different from the other two --

- AI Agent top hit: `Prime Agent: A Self-Improving RLM Harness` (+0.678)
- AI coding top hit: `Claude Code costs up to $200 a month. Goose does the
  same thing for free.` (+0.646)
- Large Language Model top hit: `Advancing voice intelligence with new
  models in the API` (+0.627)

`RELEVANCE_KEEP_MAX=50` engaged correctly for all three (999 in, 50 kept,
matching `round(999*0.10)=100` clamped down to 50) confirming the raw pool
is genuinely uncapped-by-count going in, not silently still bounded by the
removed `RELEVANCE_SAMPLE_SIZE`. Absolute top scores are also markedly
higher than the pre-title+summary measurement earlier in this document
("Large Language Model" topped at 0.166 there; 0.627 here) -- consistent
with, not proof of, the title+summary and generated-definition changes
compounding rather than fighting each other.

## Offbeat selection, take two: keyword rule + statistical "surprising word" detection, 2026-08-26

Live use of the shipped offbeat gate (the embedding-based "lowest
similarity to the pool's own centroid, past a relevance floor" design
above) surfaced a real quality complaint: the bottom of a topic's ranked
pool -- exactly where the offbeat pick is drawn from -- reads as
*unrelated* more often than *novel*. This is the same "farthest is not
the same as interesting" finding this document already made once
(the "Novelty works, but 'farthest' is not the same as 'interesting'"
section above) resurfacing after the whole pipeline changed underneath
it -- worth re-measuring rather than assuming the earlier fix still
holds at the new relevance-filtered pool sizes.

**Two candidate signals, explicitly scoped together**: a small constant
list of novelty-signaling keywords (`leak`, `breakthrough`, `unveils`,
`lawsuit`, `banned`, `warns`, ...) checked directly against title+summary
text, plus a statistical "this article combines vocabulary that doesn't
normally go together" detector -- the running example throughout this
investigation was "an AI-agent article that also happens to mention a
street cat." BM25 itself cannot supply the second signal: it ranks
documents against one query, with no notion of "these two things
co-occurring is unusual." That requires an actual co-occurrence-surprise
statistic, measured directly against the same 2604-article production-
cache pull (`full_cache.jsonl`) used throughout this document, all real
model2vec/NLTK, no synthetic fixtures.

**The keyword signal worked from the first test and stayed reliable
across every iteration below** -- flagged articles (a Twitch/Amazon
AI-training lawsuit, a Meta antitrust-trial backlash story, an
early-access Claude-model leak, public backlash over datacentre buildout)
consistently read as genuinely notable, not noise. The statistical half
took five real-data iterations to get right; four failed for four
different, specific, measured reasons, kept here because the sequence of
failures is itself the finding -- each one looked reasonable until it was
actually run against real data.

### Attempt 1 -- unigram PMI, within-article pairs: too noisy

For each candidate article, score every pair of its own words by
corpus-wide PMI (`log(P(a,b) / (P(a)*P(b)))`), take the pair with the
lowest score as the article's "surprise" signal. Result: the flagged
pairs were meaningless noise -- `models+people`, `based+openai`,
`across+trump`, `power+use`. Root cause: an article with even a modest
vocabulary produces dozens of candidate pairs (`C(k,2)` for `k` words),
and taking the *minimum* over that many noisy scores finds a spuriously
low value by pure multiple-comparison chance almost every time,
regardless of whether the article contains a genuinely unusual
combination.

### Attempt 2 -- bigram PMI: systematically inflated, not just noisy

Same shape, but pairing adjacent-word bigrams ("ai_agent") instead of
single words, hoping to preserve phrase-level meaning. Median score
across the whole pool: **+0.772** -- strongly *positive*, the opposite
direction from "surprising." Root cause is a well-documented PMI failure
mode: bigrams are far sparser than unigrams (most near the minimum
occurrence floor), and PMI is known to be biased toward rare events --
two low-frequency phrases that happen to co-occur even once look
"highly associated" by the formula's math, which is an artifact of
sparse data, not a real semantic signal.

### Attempt 3 -- noun-only unigrams (NLTK POS tagging): fixed the math, not the signal

Restricting to nouns (via `nltk.pos_tag`) removed the generic
verbs/prepositions (`using`, `based`, `across`) that were polluting
attempt 1. Numerically this worked -- scores landed in a sane range
(-0.19 to +0.07), no more flooring or inflation. But isolating the
statistical signal from the keyword rule (ranking articles with **no**
keyword hit purely by lowest noun-pair PMI) still surfaced generic pairs
-- `models+time`, `openai+state`, `models+tech` -- with `models+time`
independently topping two completely unrelated articles. A pair's low
PMI was still a property of the *words themselves* (how they behave
across the whole corpus), not of the *specific article* containing them.

### Attempt 4 -- + WordNet concrete/abstract filtering: coverage collapsed, and a new artifact appeared

User-directed refinement: also exclude abstract nouns (`time`, `state`,
`trade`, `control`) via WordNet's hypernym hierarchy (a noun's most
common sense classified as `physical_entity` vs. `abstraction`), keeping
words absent from WordNet entirely (proper nouns/brands like "OpenAI")
since those are exactly the entity terms worth keeping, not noise.
Directionally correct in spirit, but the real-data run was worse on both
axes that matter: only 156 of 999 pool articles still had a scorable
concrete-noun pair (vs. 551/999 for plain nouns) -- most candidates would
have had no offbeat signal at all -- and the surviving "no keyword hit"
top picks were now dominated by one single hyper-frequent term
(`openai`) pairing with almost anything (`openai+world` alone topped
three unrelated articles). A word that's simply very common in the pool
will look "unusually paired" with nearly everything, which is a property
of that word's frequency, not of any specific article.

### The actual fix: keyness against the topic pool, not pairs against each other

**User's correction, verbatim: "You should not [be] using OpenAI against
other words, but against AI agent."** Every attempt above compared two
arbitrary words *within* an article against each other. The fix compares
each of an article's own words *against the topic itself* -- specifically,
against the topic's whole category pool (e.g. "AI"), not the bare topic
phrase (which often isn't literally present in a given article's text).
This is standard corpus-linguistics **keyness** analysis (Rayson & Garside
2000): a 2x2 contingency table of "does this word appear" x "topic pool
or rest of corpus," scored with a **signed log-likelihood ratio (G2,
Dunning 1993)** rather than raw PMI -- G2 is the standard fix for PMI's
small-count instability (the exact failure in attempts 1-2), and signing
it by direction (present in-topic more or less than its own overall rate
predicts) distinguishes a topic-defining word from a genuinely foreign
one, which unsigned G2 alone cannot.

Only per-word, not per-pair: an article with `k` nouns has `k` chances to
be flagged, not `C(k,2)` -- this alone removes most of the multiple-
comparison noise the first three attempts were fighting.

**Sanity check, directly answering the user's correction**: `keyness(
"openai", AI) = +286.95`, `keyness("agent", AI) = +80.39`,
`keyness("agents", AI) = +130.57` -- all strongly *positive*, correctly
identified as topic-typical vocabulary, not flagged as foreign. The
articles that DO score as topic-foreign read as genuinely offbeat for an
"AI Agent" reader: arXiv quantum-computing papers that happen to be
AI-tagged, US political coverage that happens to be AI-tagged, Japan/
climate/geopolitics crossover pieces -- content that passed the category
filter honestly but isn't what a subscriber to that specific interest
would expect.

**Known, accepted limitation**: because the score is per-word, not
per-article, articles sharing the same flagged foreign word (e.g. eight
different quantum-physics papers all triggering on "quantum") tie
exactly. Not treated as a blocker -- `_pick_for_topic`'s existing
recency-ordered pool already breaks ties sensibly, the same way it
already does for the embedding-based gate's own ties.

**Status: shipped 2026-08-26, as `news_keyness.py`.** The embedding-based
near-duplicate collapse and relevance filter are unaffected -- only the
offbeat/novelty slot selection was replaced. WordNet-based concrete/
abstract filtering (attempt 4 above) was measured to make things worse
and is NOT part of the shipped design -- only POS tagging is used, not
the full NLTK data footprint this section's experiments pulled in.

**The "second VM" plan floated during this investigation was considered
and then reversed, once actually measured.** The initial assumption was
that NLTK might be too heavy for the bot VM's own memory budget (by
analogy to sentence-transformers/fastembed's rejection earlier in this
document), so a separate, otherwise-idle VM was set up to run keyness as
a standalone periodic batch job. Measured there: 84.6 MB peak RSS for
the whole computation over 2673 real articles -- comfortably inside the
*bot* VM's own existing headroom. Once that number existed, the second-
VM design no longer had a justification, and carried a real cost the
in-process design doesn't: a periodic-batch-plus-file-sync architecture
means a genuine staleness window between an article being ingested and
its category's keyness table reflecting it, hitting exactly the newest,
most novelty-relevant content hardest. Shipped instead as part of
`news_ingest.py`'s own cycle, same cadence and same machine as
`news_embed.py`'s embedding step, with no cross-VM sync at all. See
`docs/current/infrastructure.md` for the second VM's resulting (idle)
status.

### Product correction, same day: additive, threshold-gated, optional

The first shipped version of this feature (above) carved 1-2 of
`MAX_ARTICLES_PER_TOPIC`'s regular slots out for the novelty pick, and
always filled them with whatever scored best in the pool -- even when
nothing had a real signal (see `_offbeat_sort_key`'s neutral-tiebreak
fallback in the pre-correction code). User-directed correction, three
explicit requirements:

1. This is an experimental feature; it doesn't need to be precise --
   an occasional pick that isn't genuinely novel is an accepted cost.
2. The novelty pick belongs in the SAME push message, but as an
   **additional**, clearly separate closing section (e.g. "by the way,
   we noticed something interesting"), not blended into the main report
   and not competing with the regular candidates for space.
3. It needs an actual **threshold** -- only a candidate that clears it
   counts as novel, and a cycle where nothing clears it sends no novelty
   section at all. Forcing a pick regardless of signal strength was
   explicitly rejected.

Implemented as `news_push._pick_novelty_extra` (replacing `_pick_for_
topic`): the regular digest is always `pool[:max_per_topic]` by pure
recency, with no novelty carve-out at all; the novelty extra is drawn
separately from `pool[max_per_topic:]` (articles that passed the
relevance filter but didn't make the regular cut, so it never re-
suggests something already in the digest), and only returned when a
keyword hit is present OR the article's lowest keyness score clears
`NOVELTY_KEYNESS_THRESHOLD = -5.0` -- otherwise `None`, deliberately, not
a fallback pick. -5.0 is a first-cut chosen from this document's own
real numbers (clearly-foreign terms measured at -19 to -40 in the real
999-article "AI" pool), not fitted against real push outcomes yet --
revisit once some have accumulated. `write_push_digest`'s prompt marks
at most one candidate `[EXTRA]` in the listing and instructs the model
to format it as a separate closing note if included (still exercising
its own relevance judgment, same as any other candidate) -- deliberately
NOT the same shape as the retired per-line `[topic]` prefix (a marker on
every line reads as confirmed metadata; a marker on the one genuinely-
different item is a real, narrow formatting instruction).

### 2026-08-27 incident: an off-topic novelty pick reached a real subscriber

A subscriber with interest "robotics" (chat_id 952604638) received a
push digest whose closing note discussed a Witcher 3 game-DLC article --
and the digest's own text admitted the mismatch ("雖然與機器人學無關", "though
unrelated to robotics"), rather than silently omitting it as
`_PUSH_DIGEST_PROMPT` instructs. Root-caused via direct production
database and cache queries, a three-link chain:

1. The LLM classifier had genuinely misjudged "robotics" into zero of
   the taxonomy's 28 categories (confirmed: `interest_categories` row
   `('robotics', '[]')`) -- a real classification miss, not a system bug.
2. `select_candidate_articles` treated a topic with no mapped categories
   as **unrestricted** ("matches every article"), a design choice this
   same docstring had already flagged as dangerous after the 2026-08-17
   classification outage, but never revisited for the *confidently-
   empty* case specifically (only retryable failures got fixed then).
   Under "unrestricted," a techradar.com Witcher 3 DLC article
   (`categories=['Media','Consumer']`, summary containing "unveiled")
   became a candidate for "robotics" despite sharing nothing with it.
3. `has_novelty_keyword`'s substring match has no topic awareness at
   all -- "unveiled" hit regardless of what the article was actually
   about, and with nothing upstream gating novelty eligibility on
   relevance beyond the regular digest's own leftover pool, the article
   won the novelty slot. `write_push_digest`'s own relevance judgment
   (stage 2) was the last line of defense and it also failed, producing
   the self-contradicting output instead of omitting the article.

**Fix 1 -- reverse the fail-open "unrestricted" behavior.** A topic
with no mapped categories now gets zero candidates and is skipped
entirely for that cycle (same downstream shape as "nothing new since
last time") instead of matching anything. This alone closes the
specific incident: the Witcher 3 article could never have entered
`raw_pool` at all, since "robotics" would never reach the category-match
or novelty-search code paths in the first place. Accepted tradeoff,
not a strict improvement: a subscriber whose interest is genuinely too
novel/niche for the current 28-category taxonomy now gets nothing for
it, silently, rather than an imperfectly-filtered digest -- and because
`resolve_interest_categories` caches a genuinely-empty classification
permanently (never retried, unlike a failure), this is a standing
condition per affected interest, not a one-cycle blip. Judged the
better failure mode: a quiet gap is worse to eventually notice than an
unrelated-content leak is to receive without warning, but it's a real
cost, not a free fix.

**Fix 2 -- give novelty eligibility its own relevance floor.** Even for
a topic WITH real categories, the coarse category filter alone doesn't
guarantee topical relevance (the same gap `_filter_by_relevance` exists
to close for the regular digest) -- and until this fix, the novelty
search only ever saw `pool[max_per_topic:]`, the regular digest's own
leftover after its stricter `RELEVANCE_KEEP_*` cut, with no relevance
floor of its own beyond that. Added `NOVELTY_RELEVANCE_KEEP_FRACTION =
0.20` / `NOVELTY_RELEVANCE_KEEP_MIN = 40` / `NOVELTY_RELEVANCE_KEEP_MAX
= 100` -- roughly double the regular `RELEVANCE_KEEP_*` clamp, a
first-cut guess in the same spirit as `NOVELTY_KEYNESS_THRESHOLD`'s
-5.0, not yet tuned against real push data. `select_candidate_articles`
now runs a SECOND, separate `_filter_by_relevance` pass over `raw_pool`
(not `pool`) with this wider clamp, and feeds novelty search whatever
survives that minus the regular candidates already picked. Deliberately
wider than the regular cut, not equal to it: novelty content is allowed
to be "not so important, but eye-opening" per this feature's own design,
so requiring it to clear the exact same bar as a regular candidate
would defeat the point -- it only needs to prove "still related to
topic," not "would have made the regular digest." Covered by
`tests/test_news_push.py::test_novelty_extra_excluded_when_it_fails_
even_the_wider_relevance_bar` (a keyword-hit article with ~zero
embedding similarity to the topic, in a pool large enough for the
percentile cut to actually exclude something, is confirmed excluded)
and `::test_novelty_extra_uses_a_wider_relevance_pass_than_the_regular_
digest` (pins the two separate `_filter_by_relevance` calls' params).

## Decision 2026-11-07: no category layer in ingestion or retrieval

Pulling news no longer calls an LLM, and candidate selection no longer gates
on article categories. `news_ingest.run_ingestion_cycle` caches each article
with its embedding only (`categories` is stored as NULL); `news_push.
select_candidate_articles` matches an interest to articles by embedding
similarity alone, and the push cycle no longer classifies interests.
Offline check on the live cache (robotics): the old category gate left 20
candidates (20 Jev-approved); without it, a pool of 50 gave 45 Jev-approved
(25 more, 0 lost). Edge AI boards stayed at 6-7 either way (few relevant
articles in the cache, not a gate effect). Costs: the novelty extra now uses
keywords only (per-category keyness needs category pools), and the
category-proposal admin flow goes dormant. The `categories` column,
`news_classify` and the category tables are left in place.
