"""
Batched article classification for the local news cache -- see
docs/plans/local-news-cache-plan.md's "Classification mechanism" section.

One structured-output LLM call per ingestion cycle classifies every
newly-fetched article in that cycle at once, rather than one call per
article -- at the volumes this project's source registry produces, a
per-article call would mean hundreds of LLM calls/day for a task a single
batched call handles just as well. Same DeepSeek instance already used
elsewhere (see docs/plans/local-news-cache-plan.md's resolved "classification
model" question) -- docs/plans/model-portability-plan.md's cheaper per-stage
routing can swap this later without a redesign.

The taxonomy is no longer defined here. It lives in the `categories`
table (see category_ops.SEED_CATEGORIES for what a fresh database starts
with, and docs/plans/taxonomy-and-admin-plan.md for the design), and is
passed in as a Taxonomy so this module keeps no database dependency. It
is explicitly revisable: the whole point of moving it was that changing
it should be a data change, not a redeploy.
"""

from dataclasses import dataclass
from functools import cached_property

from pydantic import BaseModel

from telemetry import EventLogger, get_event_logger
from telemetry_providers import Level

_events: EventLogger = get_event_logger("argus.news_classify")


@dataclass(frozen=True)
class Taxonomy:
    """The categories the classifier may use, for one run.

    Passed in rather than read from the database here, so this module has
    no database dependency and its tests can hand it a two-category
    taxonomy instead of standing up SQLite -- the same reasoning as
    build_agent taking its model as a parameter (see CLAUDE.md). Callers
    build it from category_ops.get_active_categories().

    Frozen because the prompt text is derived from it: a taxonomy that
    changed underneath a batch would mean the prompt and the validation set
    disagreed about what a valid answer is.
    """

    entries: tuple[tuple[str, str], ...]   # (name, description), prompt order

    @classmethod
    def from_rows(cls, rows) -> "Taxonomy":
        return cls(entries=tuple((name, description) for name, description in rows))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.entries)

    def __contains__(self, name: object) -> bool:
        return name in self._name_set

    @cached_property
    def _name_set(self) -> frozenset[str]:
        """Built once. `entries` is frozen, so rebuilding a set per lookup
        would be pure waste -- negligible at a chunk of 50 articles, but
        there is no reason to pay it."""
        return frozenset(self.names)

    def prompt_fragment(self) -> str:
        return "\n".join(f"- {name}: {description}" for name, description in self.entries)


class ArticleCategories(BaseModel):
    index: int
    # Deliberately list[str], not list[Category]. A Literal here makes the
    # whole batch fail validation when the model returns one label outside
    # the taxonomy, discarding every correctly-classified article alongside
    # it. That happened in production on 2026-08-20: the model answered
    # "Education" for a single article and pydantic rejected the entire
    # 50-article ClassificationBatch, losing 49 good classifications.
    #
    # An LLM occasionally inventing a plausible label is ordinary behaviour,
    # not an exceptional condition, so unknown labels get filtered out after
    # parsing (see _valid_categories) rather than being allowed to fail the
    # request. The prompt still lists the active categories, so the model
    # is still aimed at them; this only changes what happens when it misses.
    categories: list[str]


class ClassificationBatch(BaseModel):
    items: list[ArticleCategories]


def _classify_prompt(taxonomy: Taxonomy) -> str:
    """Built per call from the taxonomy rather than being a module constant,
    so activating a category takes effect without a redeploy. The category
    list and the set used to validate the reply come from the same object,
    which is the point: a prompt offering categories the validator would
    then reject is a silent classification failure."""
    return (
        "You are a strict classifier, not an assistant. Below is a numbered "
        "list of news article titles (and summaries, where available). For "
        "EACH article, assign every category from this list that plausibly "
        "applies -- most articles need more than one:\n\n"
        f"{taxonomy.prompt_fragment()}\n\n"
        "Return one entry per article, using its exact index number from the "
        "list below. If nothing applies, return an empty categories list for "
        "that index rather than guessing or omitting the entry."
    )


def _format_batch(articles: list[dict]) -> str:
    lines = []
    for i, article in enumerate(articles):
        title = article.get("title") or "(no title)"
        summary = article.get("summary")
        line = f"{i}. {title}"
        if summary:
            line += f" -- {summary}"
        lines.append(line)
    return "\n".join(lines)


def classify_interests(model, interests: list[str], taxonomy: Taxonomy) -> dict[str, list[str]]:
    """Classifies subscriber-stated interest strings (e.g. "機器人科技",
    "AAOI") into the same category taxonomy as articles -- the stage-1
    filter news_push.py uses to narrow the shared cache to a subscriber's
    topics before the digest-writing model ever sees anything. Thin
    wrapper around classify_articles: each interest is treated as a
    single-line pseudo-article (title=interest text, no summary). Callers
    are expected to cache the result (see interest_cache_ops.get_cached_interest_categories/
    set_interest_categories) rather than re-classifying on every push
    cycle -- interest text is stable vocabulary, not fresh content.

    Interests the classifier failed on are OMITTED from the result rather
    than mapped to []. The caller must not cache an absent interest: an
    empty list is a real answer ("no category applies") and gets cached
    forever, while a failure has to be retried. Collapsing the two is what
    poisoned the live cache during the classification outage -- "AI" ended
    up cached as [] even though AI is itself one of the categories, and an
    empty mapping matches every article, so those subscribers were being
    sent completely unfiltered news."""
    if not interests:
        return {}
    articles = [{"title": interest, "summary": None} for interest in interests]
    result = classify_articles(model, articles, taxonomy)
    return {interest: result[i] for i, interest in enumerate(interests) if i in result}


# Articles per classification call. One call per ingestion cycle was the
# original design and it worked while a cycle produced ~100 articles; it
# stopped working when docs/plans/local-news-cache-plan.md item 7 raised the
# RSS per-source cap from 5 to 200 and cycles started producing 100-1000+.
#
# Measured from production on 2026-08-19, by grouping cached articles by the
# tick that wrote them:
#
#     batch    1  -> 100% categorized      batch  113 -> 0%
#     batch   60  ->  95% categorized      batch  147 -> 0%
#     batch  109  ->  96% categorized      batch 1085 -> 0%
#
# All-or-nothing, with the cliff somewhere just above 110. The failure is
# almost certainly the structured-output response exceeding the model's
# output token limit -- one entry per article, so response length scales
# with batch size. Whatever the precise cause, the fix is the same: keep
# each call's response small enough to complete.
#
# 50 is deliberately well under the observed cliff rather than just below
# it, since the cliff will move with prompt/model changes and the cost of
# an extra call is far lower than the cost of silently losing a batch.
MAX_ARTICLES_PER_CALL = 50


def _classify_one_batch(model, articles: list[dict], taxonomy: Taxonomy,
                        on_unknown_label=None) -> dict[int, list[str]]:
    """One structured-output call. Returns {} on any failure -- see
    classify_articles for the fail-open reasoning."""
    try:
        # method="function_calling" -- see guardrails.classify_message's
        # docstring/comment for why (DeepSeek's real API rejects
        # langchain_openai's default "json_schema" response_format).
        structured = model.with_structured_output(ClassificationBatch, method="function_calling")
        result = structured.invoke(
            [
                {"role": "system", "content": _classify_prompt(taxonomy)},
                {"role": "user", "content": _format_batch(articles)},
            ]
        )
    except Exception as exc:
        _events.log("batch_classify_failed",
                     {"message": f"batch of {len(articles)} failed",
                      "batch_size": len(articles)},
                     level=Level.WARN, exc=exc)
        return {}
    if result is None:
        print(f"[news_classify] batch of {len(articles)} returned no result")
        return {}
    return _valid_categories(result, taxonomy, articles, on_unknown_label)


def _valid_categories(result: ClassificationBatch, taxonomy: Taxonomy,
                      articles: list[dict], on_unknown_label=None
                      ) -> dict[int, list[str]]:
    """Drops labels the model invented that aren't in the taxonomy, keeping
    everything else. An article left with no valid labels keeps an empty
    list -- same as the model saying nothing applies.

    Rejected labels are reported rather than silently discarded, because
    they are useful signal in their own right: a label the model keeps
    reaching for is a gap in the taxonomy. "Education" is what surfaced
    this. See docs/plans/taxonomy-and-admin-plan.md, where these labels
    become the evidence an admin decides on.

    `on_unknown_label(label, article)` is how they leave this module. A
    callback rather than a database write here, and rather than a richer
    return type, for two reasons: it keeps this module free of a database
    dependency (the same reason Taxonomy is a parameter), and it carries
    the example article along, which the admin prompt needs -- a bare label
    with no example is much harder to make a decision about.

    The first article seen for a label is passed, not all of them: this
    runs per 50-article chunk many times a day, and the admin prompt shows
    two or three examples, so accumulating every one would grow a table
    nobody reads to the bottom of."""
    categories: dict[int, list[str]] = {}
    # {rejected label: best example article so far}. One structure rather
    # than a set for dedup plus a dict for examples: keeping "have we seen
    # this label" and "do we have an example for it" in the same key made
    # whichever entry mentioned a label FIRST win, so a malformed entry
    # (index past the end, no article) arriving before a good one blocked
    # the good one. Insertion-ordered, so reporting order is stable.
    unknown: dict[str, dict] = {}
    for item in result.items:
        kept = []
        # index is the model's, so a malformed reply can point past the end
        # of the chunk. That must not lose the batch.
        article = articles[item.index] if 0 <= item.index < len(articles) else None
        for name in item.categories:
            if name in taxonomy:
                kept.append(name)
                continue
            if article is not None and not unknown.get(name):
                unknown[name] = article
            else:
                unknown.setdefault(name, {})
        categories[item.index] = kept
    if unknown:
        print(f"[news_classify] dropped {len(unknown)} label(s) outside the "
              f"taxonomy: {', '.join(sorted(unknown))}")
        if on_unknown_label is not None:
            for name, example in unknown.items():
                # An empty example rather than skipping: a sighting with no
                # article attached is still evidence, and losing it would be
                # worse than showing the admin a bare label.
                on_unknown_label(name, example)
    return categories


def classify_articles(model, articles: list[dict], taxonomy: Taxonomy,
                      on_unknown_label=None) -> dict[int, list[str]]:
    """Returns {index: categories} for every article in `articles` that the
    model actually returned an entry for, where the index is into
    `articles` as given.

    Split into chunks of MAX_ARTICLES_PER_CALL rather than one call for the
    whole cycle -- see that constant for the production measurement that
    forced it. Chunking also bounds the blast radius: a failed call now
    costs one chunk's categories instead of the entire cycle's.

    Fails open on any error (model call failure, malformed response) by
    omitting that chunk's indexes. A missing index means UNCLASSIFIED --
    nothing is known about that article -- which callers must keep distinct
    from an index that IS present with an empty list, meaning the model
    looked and nothing applied. news_ingest records the first as null and
    the second as category_ops.UNCLASSIFIABLE; collapsing them is what made a
    three-day outage look like normal operation. Same
    reasoning as guardrails.classify_message: a classification hiccup
    shouldn't block caching the article, it should just make it harder to
    find via category filtering until a later cycle reclassifies it.

    Unlike before, a failure is now *printed*. The silent version hid a
    real outage: every batch above ~110 articles failed for three days
    straight, leaving 92.8% of the cache uncategorized and therefore
    invisible to push candidate selection, with nothing in the logs to
    show for it."""
    if not articles:
        return {}
    categories: dict[int, list[str]] = {}
    failed_chunks = 0
    total_chunks = 0
    for start in range(0, len(articles), MAX_ARTICLES_PER_CALL):
        chunk = articles[start:start + MAX_ARTICLES_PER_CALL]
        total_chunks += 1
        result = _classify_one_batch(model, chunk, taxonomy, on_unknown_label)
        if not result:
            failed_chunks += 1
        for local_index, cats in result.items():
            # translate the chunk-local index the model returned back to an
            # index into the caller's own list
            if 0 <= local_index < len(chunk):
                categories[start + local_index] = cats
    if failed_chunks:
        print(
            f"[news_classify] {failed_chunks} of {total_chunks} chunk(s) failed -- "
            f"{len(articles) - len(categories)} article(s) left UNCLASSIFIED "
            f"(nothing known about them -- not the same as having no category)"
        )
    return categories


class CategoryDescription(BaseModel):
    # Reasoning first so the model works before it commits, the same
    # field-order lesson already applied in guardrails.OutputCheck.
    reasoning: str
    description: str


def draft_category_description(model, name: str, examples: list[str],
                               existing: Taxonomy) -> str | None:
    """Drafts the one-line gloss for a proposed category, from the articles
    that actually triggered it.

    This text goes into the classifier prompt verbatim for every article
    afterwards, so a vague one silently degrades classification from then
    on. Drafting it here means the admin sees the exact wording in the
    approval message and can judge it, rather than facing a blank field
    after they have already committed to adding the category.

    The existing taxonomy is shown to the model so the description can say
    what this category is NOT -- the failure being avoided is a new
    category that overlaps an old one, which makes the classifier
    inconsistent between them rather than wrong in any one place. Policy
    was exactly that: its description listed four things and it absorbed
    65% of all assignments on a general-news probe.

    Returns None on failure; the caller falls back to asking the admin,
    since a missing description is recoverable and a wrong one is not."""
    try:
        # method="function_calling" -- see guardrails.classify_message's
        # docstring/comment for why.
        structured = model.with_structured_output(CategoryDescription, method="function_calling")
        result = structured.invoke([
            {"role": "system", "content":
                "You write one-line category descriptions for a news classifier. "
                "The description is shown to the classifier as its only guidance "
                "for the category, so it must be concrete and must not overlap "
                "the existing categories. Match the style of the existing ones: "
                "lowercase, a short comma-separated list of what belongs, no "
                "trailing period. Where there is a risk of confusion with an "
                "existing category, say what this one is NOT."},
            {"role": "user", "content":
                f"New category: {name}\n\n"
                f"Articles the classifier used it for:\n"
                + "\n".join(f"- {t}" for t in examples)
                + f"\n\nExisting categories:\n{existing.prompt_fragment()}"},
        ])
    except Exception as exc:
        _events.log("description_draft_failed",
                     {"message": f"could not draft a description for {name!r}",
                      "name": name},
                     level=Level.WARN, exc=exc)
        return None
    if result is None or not result.description.strip():
        return None
    return result.description.strip()


def _interest_request(text: str, alongside: list[str] | None) -> str:
    if not alongside:
        return f"Interest: {text}"
    return (
        f"Interest: {text}\n\n"
        f"The same subscriber also follows: {', '.join(alongside)}.\n"
        "Use this to disambiguate an ambiguous abbreviation or ticker -- "
        "pick the reading that fits what they already follow."
    )


class NormalizedInterest(BaseModel):
    # Reasoning first so the model commits after working, not before --
    # the field-order lesson from guardrails.OutputCheck.
    reasoning: str
    english: str
    # An explicit judgment, made BEFORE the examples, because asking only
    # for "narrower readings" gets them for everything -- almost any phrase
    # has a narrower phrase. Measured against the live model on the first
    # attempt: "Local LLM", "AI Agent" and "Optical communications" all
    # came back with suggestions, which would have made the hint fire on
    # nearly every interest and taught subscribers to ignore it.
    is_umbrella: bool = False
    # Empty when the interest is already specific enough to retrieve well.
    # Non-empty means it is a broad umbrella, and these are concrete
    # narrower readings of it -- used only to HINT, never to block or to
    # ask a follow-up question, so an over-eager model costs a sentence
    # rather than an interrogation.
    #
    # It rides along on the normalization call that already happens for
    # every added interest, so the whole feature costs no extra request.
    narrower_examples: list[str] = []


def normalize_interest_detailed(model, text: str, alongside: list[str] | None = None) -> "NormalizedInterest | None":
    """Turns a subscriber's stated interest into a short ENGLISH label.

    Interest text is not just displayed -- it is used as a live search
    query against every query-capable source, matched lexically by BM25,
    and classified into categories. All three of those are English-facing,
    so a non-English interest degrades quietly rather than failing:

      - `GNewsAdapter.pull` pins `lang=en` and `NewsApiAdapter.pull` now does too, so
        a Chinese query returns nothing at all. Measured: 0 articles for
        機器人科技 and 光通訊, 10 for "robotics".
      - BM25 scored **0%** recall for 光通訊 against this corpus, because
        no English article shares a token with it. Not weak -- structurally
        incapable. Embeddings managed 57%, which is what hid the problem.
      - A bare ticker is worse still: "AAOI" retrieves 0/30 relevant
        articles by BOTH methods, since the corpus contains no such token
        and four letters carry no semantics. "Applied Optoelectronics"
        gets 6/30 by embedding. So a ticker is expanded rather than
        preserved alone -- the ticker is the lexical handle BM25 needs when
        an article does mention it, the company name is the semantic handle
        the embedding needs, and dropping either loses one retrieval path.
      - The corpus is 97.6% English, so any non-English text is an outlier
        by script alone, which distorts both clustering and the
        farthest-from-everything novelty pick.

    Stores English rather than keeping the original because every consumer
    is English-facing. The subscriber still gets confirmations in their own
    language (bot.py translates Route B replies), and seeing the English
    label in /interests tells them how the system actually understood
    them -- which is worth knowing when it got it wrong.

    `alongside` is the subscriber's OTHER interests, used to disambiguate.
    Expanding a ticker without context picks the wrong company and is then
    worse than not expanding at all: "AOI" alone came back as "Africa Oil
    Corp", which would drag retrieval toward oil news, when the subscriber
    who stored it also tracks AAOI, semiconductors and optical
    communications and plainly meant automated optical inspection. The
    disambiguating information is already in the database and costs
    nothing to include.

    Returns None on failure; the caller keeps the original text rather than
    dropping the interest, since a stored interest that searches badly
    beats an interest that silently wasn't saved."""
    if not text.strip():
        return None
    try:
        # method="function_calling" -- see guardrails.classify_message's
        # docstring/comment for why.
        structured = model.with_structured_output(NormalizedInterest, method="function_calling")
        result = structured.invoke([
            {"role": "system", "content":
                "You normalize a news-subscription interest into a short "
                "English label of 2-4 words. Translate if it isn't English. "
                "For a stock ticker, keep the ticker AND add the company "
                "name: \"AAOI\" becomes \"AAOI Applied Optoelectronics\". "
                "Prefer the term the industry press would actually use in a "
                "headline over a literal translation. No punctuation, no "
                "explanation. "
                "Also set is_umbrella. It is true ONLY for a whole "
                "field or industry, the kind of word a newspaper "
                "would name a section after: AI, tech, crypto, "
                "software, hardware, robotics, biotech. It is FALSE "
                "for anything that already names a specific "
                "technology, product, company, ticker or research "
                "area, even when a still-narrower phrase exists: AI "
                "Agent, Local LLM, optical communications, quantum "
                "sensing, AAOI and RISC-V are all false. When in "
                "doubt, false. Only if is_umbrella is true, put 2-4 "
                "concrete narrower readings in narrower_examples; "
                "otherwise leave it empty. Never narrow it yourself "
                "-- english stays the faithful label of what they "
                "actually said."},
            {"role": "user", "content": _interest_request(text, alongside)},
        ])
    except Exception as exc:
        _events.log("interest_normalize_failed",
                     {"message": f"could not normalize interest {text!r}",
                      "text": text},
                     level=Level.WARN, exc=exc)
        return None
    if result is None or not result.english.strip():
        return None
    return result


def normalize_interest(model, text: str, alongside: list[str] | None = None) -> str | None:
    """The English label alone -- see normalize_interest_detailed, which
    this wraps, for what the call actually does and why.

    Kept as its own function because most callers want a string and
    nothing else; only the add-an-interest path has any use for the
    breadth hint."""
    result = normalize_interest_detailed(model, text, alongside)
    return result.english.strip() if result else None


class _RetrievalExpansion(BaseModel):
    # Reasoning first -- same field-order lesson as everywhere else in
    # this module (guardrails.OutputCheck, NormalizedInterest).
    reasoning: str
    definition: str


def expand_interest_for_retrieval(model, interest: str) -> str | None:
    """A short ENGLISH definition of `interest`, used as the embedding
    query in news_push.select_candidate_articles's relevance filter and
    offbeat gate INSTEAD OF the bare interest string.

    Measured, 2026-08-25: embedding the bare phrase "AI coding" needed
    the relevance filter to keep 83% of a topic's candidate pool to avoid
    losing any of 5 genuinely relevant real articles -- three "Claude
    Cowork" pieces scored 0.18-0.19 cosine similarity against that
    phrase, BELOW an unrelated stock-picking article at 0.32, because
    model2vec's static embeddings have no way to connect a product name
    to a category word without shared vocabulary. Embedding a generated
    definition instead -- one that names concrete tools, techniques and
    related terms an article on the subject would actually use --
    dropped the same worst-case to 44%. See
    docs/analysis/cluster-measurements.md's "Shipped" section for the
    full measurement, including why a longer generated article OUTLINE
    (HyDE-style) barely improved on a plain definition (41% vs 44%) --
    most of the gain is in having ANY topic-specific vocabulary at all,
    not in how much of it.

    Called once per NEWLY-SEEN interest string (agent.py's
    add_one_interest, cached in interest_cache_ops's global
    interest_query_expansions table, the same shape and same reasoning
    as resolve_interest_categories's cache: the interest text is stable
    vocabulary, so this should be a cache hit for any interest that's
    been added by any subscriber before). Never called on the hot push
    path -- see news_push.py's _resolve_query_text, which reads the
    cache and falls back to the bare topic string when nothing is
    cached, e.g. an interest added before this feature existed.

    English regardless of the interest's own language or the
    subscriber's reply-language preference -- same reasoning as
    normalize_interest_detailed: retrieval is English-facing throughout
    this pipeline, and a Chinese query was measured (same session) to
    perform WORSE than even the bare English phrase against this
    English-dominant corpus (66% needed vs 44%), because model2vec's
    training skews English and a non-English query shares almost no
    subword vocabulary with the corpus it's compared against.

    Returns None on failure or a blank result -- the caller falls back
    to the bare topic string, which is a worse retrieval signal but a
    real one, not a missing interest."""
    if not interest.strip():
        return None
    try:
        # method="function_calling" -- see guardrails.classify_message's
        # docstring/comment for why.
        structured = model.with_structured_output(_RetrievalExpansion, method="function_calling")
        result = structured.invoke([
            {"role": "system", "content":
                "Write a short ENGLISH definition of the given tech-industry "
                "news interest, 2-4 sentences, for use as a semantic search "
                "query -- not for a human to read. Name concrete tools, "
                "products, techniques, or related terms a real news article "
                "on this subject would actually use, the way a glossary "
                "entry or a Wikipedia lead paragraph would, not a vague "
                "restatement of the interest itself. Translate first if the "
                "interest isn't already in English. If the interest is a "
                "company, product, or ticker name rather than a general "
                "topic, describe what it does and its category instead of "
                "just repeating the name."},
            {"role": "user", "content": f"Interest: {interest}"},
        ])
    except Exception as exc:
        _events.log("interest_expand_failed",
                     {"message": f"could not expand interest {interest!r} for retrieval",
                      "interest": interest},
                     level=Level.WARN, exc=exc)
        return None
    if result is None or not result.definition.strip():
        return None
    return result.definition.strip()
