"""
Test doubles for the LLM and telemetry/logging layers.

Neither LangChain's built-in fakes (GenericFakeChatModel,
FakeMessagesListChatModel) implement bind_tools() — langchain.agents.
create_agent calls model.bind_tools(tools, tool_choice=...) unconditionally
when tools are present, so those fakes raise NotImplementedError. Confirmed
live before writing this. FakeToolCallingModel below overrides bind_tools()
to return self, ignoring the schema, since responses are pre-scripted.
"""

import hashlib
import re

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolCallingModel(BaseChatModel):
    """Cycles through a scripted list of AIMessage responses (stays on the
    last one once exhausted). Pass tool_calls on an AIMessage to script a
    tool call; create_agent executes the real tool and feeds the result
    back for the next response in the list."""

    responses: list[AIMessage]
    i: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = self.responses[self.i]
        self.i = min(self.i + 1, len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-model"


class FakeSpan:
    """Stand-in for an OTel span, for asserting on EventLogger.log(...)
    calls (telemetry.py) -- monkeypatch a module's
    `<logger>._tracer.start_as_current_span` to `lambda name: FakeSpan()`
    and check `.attrs`/`.exceptions` after. Shared here because six
    call-site test files (news_embed.py, message_archive.py,
    news_ingest.py, bot.py, news_classify.py, guardrails.py) need the
    identical class -- keeping one copy means EventLogger.log() growing
    a new span method only needs updating here, not in lockstep across
    every test file that imports it (grep this file's own name across
    tests/ for the current list -- deliberately not enumerated here,
    since a hardcoded count/list just goes stale as new consumers are
    added, as happened to this docstring's own count more than once)."""

    def __init__(self):
        self.attrs = {}
        self.exceptions = []
        self.status = None

    def set_attribute(self, k, v):
        self.attrs[k] = v

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class RecordingCallbackHandler(BaseCallbackHandler):
    """In-memory stand-in for a real telemetry backend. Records LLM and tool
    start/end events as plain dicts in self.events, in call order, for tests
    to assert against instead of hitting Phoenix/LangSmith/etc."""

    def __init__(self):
        self.events: list[dict] = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.events.append({"type": "llm_start"})

    def on_llm_end(self, response, **kwargs):
        self.events.append({"type": "llm_end"})

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.events.append({"type": "tool_start", "tool": serialized.get("name"), "input": input_str})

    def on_tool_end(self, output, **kwargs):
        self.events.append({"type": "tool_end", "output": str(output)})


class FakeEmbedder:
    """Deterministic stand-in for model2vec's StaticModel -- same
    `.encode(list[str]) -> ndarray of L2-normalized rows` interface, no
    real model load, no network. Hashes each word into a fixed-size space
    and sums, so cosine similarity tracks WORD OVERLAP: "Nvidia launches
    new GPU" and "Nvidia unveils new GPU" land close together, "Bitcoin
    price surges" lands far from both. Real enough to exercise near-
    duplicate collapse and offbeat gate/rank logic meaningfully with
    ordinary English test fixtures, rather than requiring hand-crafted
    vectors for every case.

    Never call this with an empty text list -- exactly like the real
    model2vec, it raises rather than silently degrading, so a test
    exercising news_embed.embed_texts's own empty-input guard still needs
    that guard to be real."""

    DIM = 32

    def encode(self, texts: list[str]) -> "np.ndarray":
        if not texts:
            raise ValueError("need at least one array to concatenate")
        vectors = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = int(hashlib.sha256(word.encode()).hexdigest(), 16)
                dim = digest % self.DIM
                sign = 1.0 if (digest // self.DIM) % 2 == 0 else -1.0
                vectors[i, dim] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # an all-punctuation/empty string stays the zero vector
        return vectors / norms


def fake_word_tokenize(text: str) -> list[str]:
    """Stand-in for nltk.tokenize.word_tokenize -- plain alphanumeric
    splitting, no real tokenization rules (contractions, punctuation
    edge cases, etc.). Matches the function's shape (str -> list[str]),
    not its linguistic behavior; news_keyness.py's tests don't need the
    real thing, just something deterministic."""
    return re.findall(r"[A-Za-z0-9]+", text)


def fake_pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    """Stand-in for nltk.tag.pos_tag -- tags every token NN (a plain
    singular noun). news_keyness.py only checks tag membership in
    NOUN_TAGS, so real part-of-speech accuracy isn't needed for a test
    fixture; every word in a fixture's title/summary counting as a
    "noun" keeps fixtures simple (no need to pick words a real tagger
    would actually classify as nouns to get predictable test behavior)."""
    return [(t, "NN") for t in tokens]
