"""The review pipeline must emit readable spans for its LLM call.

Before this, `ReviewPipeline.__call__` returned a single word ("accept" /
"reject") and nothing else survived: the harness records the program's output
on one root span, so the review the LLM actually wrote — the THOUGHT, the
per-dimension ratings, the Overall score — was computed and discarded. The
architect had no way to see WHY a paper was rejected, and a real run shows the
cost: it rediscovered the gap in iterations 1, 2, 5, 6 and 11.

These tests pin the three properties that matter and are easy to lose while
editing prompts or the hang guard:
  1. an `llm`-kind span exists, carrying the request and the FULL response;
  2. the extracted per-dimension ratings are recorded somewhere readable;
  3. span content stays bounded — a ~50k-token paper must not be copied in.

Run with: python -m pytest tests/test_tracing.py
Needs `opentelemetry-sdk` (the harness installs it into every client venv).
"""

from __future__ import annotations

import json
import sys
import types

import pytest

REVIEW = {
    "Summary": "a summary",
    "Soundness": 3,
    "Significance": 2,
    "Overall": 6,
    "Confidence": 4,
    "Decision": "Accept",
}
RESPONSE = (
    "THOUGHT:\nThe method is sound but the evaluation is thin.\n\n"
    "REVIEW JSON:\n```json\n" + json.dumps(REVIEW) + "\n```"
)


def _install_fake_litellm(monkeypatch, *, fail_times: int = 0, response: str = RESPONSE):
    """Replace litellm with a stub that streams `response` in small chunks.

    `fail_times` makes the first N attempts raise, exercising the hang guard's
    retry path — the failure mode this program is most exposed to.
    """
    calls = {"n": 0}

    class _Delta:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.delta = _Delta(c)

    class _Chunk:
        def __init__(self, c):
            self.choices = [_Choice(c)]

    def completion(**kw):
        calls["n"] += 1
        # The seed's provider wiring and hang guard are load-bearing; the
        # additional_instructions forbid changing them, so assert they survive.
        assert kw["reasoning_effort"] == "high"
        assert kw["stream"] is True
        assert kw["timeout"] == 240
        assert kw["api_base"] == "https://api.gmi-serving.com/v1"
        if calls["n"] <= fail_times:
            raise RuntimeError(f"simulated hang #{calls['n']}")
        return [_Chunk(response[i : i + 40]) for i in range(0, len(response), 40)]

    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return calls


def _run_and_collect(monkeypatch, paper_text: str, **stub_kwargs):
    """Run the pipeline under a real tracer; return (output, spans)."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _install_fake_litellm(monkeypatch, **stub_kwargs)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The global provider can only be set once per process; when another test
    # already installed one, attach to it instead.
    existing = trace.get_tracer_provider()
    if hasattr(existing, "add_span_processor"):
        existing.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        trace.set_tracer_provider(provider)

    for mod in [m for m in sys.modules if m.endswith("review_pipeline")]:
        del sys.modules[mod]
    from src.program.review_pipeline import ReviewPipeline

    out = ReviewPipeline()(paper_text=paper_text)
    return out, list(exporter.get_finished_spans())


def _by_name(spans, name):
    return next((s for s in spans if s.name == name), None)


def test_llm_call_emits_an_llm_span_with_the_full_response(monkeypatch):
    out, spans = _run_and_collect(monkeypatch, "a short paper")

    assert out == "accept", "the decision contract must not change"
    llm = _by_name(spans, "review_llm")
    assert llm is not None, "the LLM call produced no span"

    attrs = dict(llm.attributes)
    assert attrs["ce.span_kind"] == "llm"
    assert attrs["ce.output"] == RESPONSE, "the response must be recorded in full"
    assert "THOUGHT" in attrs["ce.output"], "the model's reasoning must be readable"
    assert attrs["ce.inputs.model"] == "openai/deepseek-ai/DeepSeek-V4-Flash"
    assert attrs["ce.inputs.reasoning_effort"] == "high"
    assert attrs["attempts"] == 1
    assert attrs["stream_chunks"] > 1, "streaming must be visible"


def test_per_dimension_ratings_are_recorded(monkeypatch):
    """The ratings exist only inside the response; __call__ returns one word."""
    _, spans = _run_and_collect(monkeypatch, "a short paper")

    extract = _by_name(spans, "extract_review")
    assert extract is not None
    scores = json.loads(dict(extract.attributes)["ce.output"])
    assert scores["Soundness"] == 3
    assert scores["Significance"] == 2
    assert scores["Overall"] == 6
    assert scores["Decision"] == "Accept"


def test_retry_after_a_hang_is_visible_in_the_span(monkeypatch):
    """A recovered hang used to leave no trace at all."""
    out, spans = _run_and_collect(monkeypatch, "a short paper", fail_times=1)

    assert out == "accept", "a recovered retry must still produce a decision"
    attrs = dict(_by_name(spans, "review_llm").attributes)
    assert attrs["attempts"] == 2
    assert "simulated hang #1" in attrs["recovered_after_errors"]


def test_total_failure_records_every_attempt_error(monkeypatch):
    out, spans = _run_and_collect(monkeypatch, "a short paper", fail_times=99)

    assert out == "", "an exhausted call returns the empty decision"
    attrs = dict(_by_name(spans, "review_llm").attributes)
    assert attrs["attempts"] == 2, "MAX_ATTEMPTS attempts must be recorded"
    assert "simulated hang" in attrs["ce.error"]


def test_malformed_response_is_recorded_as_an_extraction_error(monkeypatch):
    """Malformed decisions score 0 and are never excluded from the denominator,
    so distinguishing them from a provider failure matters."""
    out, spans = _run_and_collect(
        monkeypatch, "a short paper", response="I decline to produce JSON."
    )

    assert out == ""
    attrs = dict(_by_name(spans, "extract_review").attributes)
    assert "could not extract Decision" in attrs["ce.error"]
    assert "decline" in attrs["ce.output"], "the unusable response must be inspectable"
    # The call itself SUCCEEDED — the failure was in parsing. Keeping those
    # two distinguishable is the point: one is a provider problem, the other
    # is the rubric producing unusable output.
    llm_attrs = dict(_by_name(spans, "review_llm").attributes)
    assert "ce.error" not in llm_attrs
    assert llm_attrs["ce.output"] == "I decline to produce JSON."


def test_span_content_stays_bounded_for_a_full_length_paper(monkeypatch):
    """Papers run to ~50k tokens. The row's full input is already in row.json,
    so copying it into a span would multiply trace size for no new information
    — and these traces are read by an agent with a context budget."""
    paper = "PAPER BODY " * 5000  # ~55k chars
    _, spans = _run_and_collect(monkeypatch, paper)

    llm = _by_name(spans, "review_llm")
    payload = json.dumps({k: str(v) for k, v in llm.attributes.items()})
    assert len(payload) < 20_000, f"span payload grew to {len(payload)} bytes"
    assert "PAPER BODY PAPER BODY PAPER BODY PAPER BODY" not in payload[2000:], (
        "the paper body must not be copied wholesale into the span"
    )
    messages = json.loads(dict(llm.attributes)["ce.inputs.messages"])
    user = next(m for m in messages if m["role"] == "user")
    assert user["chars"] > 50_000, "the true prompt length must still be reported"
    assert "elided" in user["content"], "the excerpt must say it was truncated"


@pytest.mark.parametrize("has_otel", [False])
def test_pipeline_works_without_opentelemetry(monkeypatch, has_otel):
    """Running this file outside the harness venv must not fail."""
    _install_fake_litellm(monkeypatch)
    for mod in [m for m in sys.modules if m.endswith("review_pipeline")]:
        del sys.modules[mod]
    import src.program.review_pipeline as rp

    monkeypatch.setattr(rp, "_otel_trace", None)
    assert rp.ReviewPipeline()(paper_text="a short paper") == "accept"
