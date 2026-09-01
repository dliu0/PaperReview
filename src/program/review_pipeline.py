# Seed program: LLM paper reviewer producing an Accept/Reject decision.
# Prompt adapted from the meta-hyperagents paper_review domain (itself adapted
# from SakanaAI/AI-Scientist perform_review.py).

import contextlib
import json
import os
import re
import sys
import time

import litellm

# --- Tracing -----------------------------------------------------------------
# The harness collects OpenTelemetry spans per row and writes them to
# `rows/<example_id>/span_NNN.json`, but only if the program emits any. Nothing
# auto-instruments a raw `litellm.completion` call, so before this the pipeline
# produced exactly ONE span carrying its final output — the single word
# "accept"/"reject" this class returns — and the review itself (the THOUGHT, the
# per-dimension ratings, the Overall score) was computed and thrown away
# unseen. The architect cannot improve a rubric whose output it cannot read.
#
# `opentelemetry-api` is installed into every client venv by the engine, so
# this needs no new requirement; it degrades to a no-op when the module is
# absent (running this file outside the harness) or when no provider is
# installed. The engine calls `setup_tracer()` AFTER importing this module, so
# the tracer is fetched lazily — OTel's proxy tracer resolves to whatever
# provider is installed by the time a span is started.
#
# Content is bounded deliberately. The row's full input is already stored in
# `row.json`, and the harness truncates the root span's copy to 2000 chars, so
# repeating a ~50k-token paper inside a span would multiply trace size for no
# new information. Messages are recorded as role + length + a bounded excerpt,
# which also survives the prompt being rewritten.
try:
    from opentelemetry import trace as _otel_trace
except ImportError:  # pragma: no cover - only outside the harness venv
    _otel_trace = None

MSG_EXCERPT_CHARS = 1500
RESPONSE_MAX_CHARS = 20000


@contextlib.contextmanager
def _span(name: str, kind: str):
    """Start a span named `name` with `ce.span_kind = kind`, or do nothing.

    Yields a setter `(key, value)` that writes a `ce.*` attribute, so callers
    need no OTel-vs-no-op branching. Never raises: a tracing failure must not
    fail a row.
    """
    if _otel_trace is None:
        yield lambda key, value: None
        return
    tracer = _otel_trace.get_tracer("paperreview.review_pipeline")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("ce.span_kind", kind)

        def _set(key: str, value) -> None:
            try:
                span.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
            except Exception:  # noqa: BLE001
                pass

        yield _set


def _summarize_messages(messages) -> str:
    """`[{role, chars, content}]` as JSON, each content bounded.

    Generic over the message list on purpose: the review form and system
    prompt are what evolves, and this keeps working whatever they become.
    """
    out = []
    for m in messages or []:
        content = str(m.get("content") or "")
        excerpt = content[:MSG_EXCERPT_CHARS]
        if len(content) > MSG_EXCERPT_CHARS:
            excerpt += f"\n... [{len(content) - MSG_EXCERPT_CHARS} more chars elided; the row's full input is in row.json]"
        out.append({"role": m.get("role"), "chars": len(content), "content": excerpt})
    return json.dumps(out, indent=2)


REVIEW_SCORE_FIELDS = (
    "Originality", "Quality", "Clarity", "Significance", "Soundness",
    "Presentation", "Contribution", "Overall", "Confidence",
    "Ethical Concerns", "Decision",
)

REVIEWER_SYSTEM_PROMPT = (
    "You are an AI researcher who is reviewing a paper that was submitted to a "
    "prestigious ML venue. Be critical and cautious in your decision."
)

NEURIPS_FORM = """
## Review Form
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.

1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
  - Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions:
  - Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited?
  - Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
  - Clarity: Is the submission clearly written? Is it well organized? Does it adequately inform the reader?
  - Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way?

2. Questions: Please list up and carefully describe any questions and suggestions for the authors.

3. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work?

4. Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review.

5. Soundness: 4: excellent / 3: good / 2: fair / 1: poor

6. Presentation: 4: excellent / 3: good / 2: fair / 1: poor

7. Contribution: 4: excellent / 3: good / 2: fair / 1: poor

8. Overall: 1-10 (very strong reject to award quality)

9. Confidence: 1-5

Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
This JSON will be automatically parsed, so ensure the format is precise.
"""


def _extract_last_json(text: str):
    """Extract the last parseable JSON object from the response text."""
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not candidates:
        # Fall back to any brace-balanced object, scanning from the end.
        starts = [m.start() for m in re.finditer(r"\{", text)]
        for s in reversed(starts):
            depth = 0
            for i in range(s, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[s : i + 1])
                        break
            if candidates:
                break
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


class ReviewPipeline:
    """One LLM call per paper: full NeurIPS-form review, then extract Decision."""

    def __init__(self):
        # Solver LM: DeepSeek-V4-Flash on GMI Cloud (reasoning=high set on the
        # call below). Same wiring as the arc-agi experiment: the optimizer
        # (GLM-5.2) and this solver both run on GMI, so we pass the GMI endpoint
        # + key explicitly (the proven GMI-as-OpenAI pattern) rather than
        # relying on OPENAI_* env, which the server points at GMI for the
        # reflection LM.
        self.model = "openai/deepseek-ai/DeepSeek-V4-Flash"
        self.api_base = "https://api.gmi-serving.com/v1"
        self.api_key = os.environ.get("GMI_CLOUD_API_KEY") or os.environ.get("GMI_API_KEY")

    # --- Hang guard (mirrors the arc-agi seed solver) --------------------
    # GMI can "hang" a request: zero bytes until its gateway kills the
    # connection minutes later, which walls an entire parallel eval at the
    # straggler. Streaming makes hangs detectable — GMI streams reasoning
    # deltas continuously (measured inter-chunk gap ~3s), so READ_GAP_TIMEOUT_S
    # of total silence is an unambiguous hang: abort fast and retry instead of
    # waiting for the gateway. httpx applies `timeout` per READ on a stream, so
    # long generations are unaffected; TOTAL_BUDGET_S caps the row across all
    # attempts.
    READ_GAP_TIMEOUT_S = 240
    TOTAL_BUDGET_S = 2400
    MAX_ATTEMPTS = 2

    def _complete(self, messages):
        """Streaming completion with hang detection; returns content text.

        Emits one `llm` span per call recording the request, the FULL response
        text and how the hang guard behaved (attempts, stream chunks, elapsed,
        the error of every failed attempt). Retries are the failure mode this
        program is most exposed to, and they were previously invisible.
        """
        start = time.monotonic()
        last_err = None
        attempts = 0
        chunks = 0
        with _span("review_llm", "llm") as set_attr:
            set_attr("ce.inputs.messages", _summarize_messages(messages))
            set_attr("ce.inputs.model", self.model)
            set_attr("ce.inputs.api_base", self.api_base)
            set_attr("ce.inputs.reasoning_effort", "high")
            set_attr("gen_ai.request.model", self.model)
            errors = []
            for _attempt in range(self.MAX_ATTEMPTS):
                if time.monotonic() - start > self.TOTAL_BUDGET_S - self.READ_GAP_TIMEOUT_S:
                    errors.append("skipped attempt: too little of the total budget left")
                    break
                attempts += 1
                try:
                    stream = litellm.completion(
                        model=self.model,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        messages=messages,
                        reasoning_effort="high",
                        allowed_openai_params=["reasoning_effort"],
                        stream=True,
                        # Per-read gap cap on a stream (NOT total duration): only
                        # trips when the connection goes fully silent (real hang).
                        timeout=self.READ_GAP_TIMEOUT_S,
                    )
                    parts = []
                    for chunk in stream:
                        if time.monotonic() - start > self.TOTAL_BUDGET_S:
                            raise TimeoutError(
                                f"row exceeded total budget {self.TOTAL_BUDGET_S}s"
                            )
                        if chunk.choices:
                            chunks += 1
                            delta = chunk.choices[0].delta
                            if delta is not None and getattr(delta, "content", None):
                                parts.append(delta.content)
                    content = "".join(parts)
                    set_attr("ce.output", content[:RESPONSE_MAX_CHARS])
                    set_attr("response_chars", len(content))
                    set_attr("attempts", attempts)
                    set_attr("stream_chunks", chunks)
                    set_attr("elapsed_s", round(time.monotonic() - start, 1))
                    if errors:
                        set_attr("recovered_after_errors", json.dumps(errors))
                    return content
                except Exception as exc:  # noqa: BLE001 — hang/gap/transient
                    last_err = exc
                    errors.append(f"{type(exc).__name__}: {exc}")
            set_attr("attempts", attempts)
            set_attr("stream_chunks", chunks)
            set_attr("elapsed_s", round(time.monotonic() - start, 1))
            set_attr("ce.error", json.dumps(errors) if errors else "completion failed")
        raise last_err if last_err else RuntimeError("completion failed")

    def __call__(self, paper_text: str = "", **kwargs) -> str:
        prompt = (
            NEURIPS_FORM
            + f"""
Here is the paper you are asked to review:
```
{paper_text}
```"""
        )
        try:
            content = self._complete(
                [
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            print(f"[review] LLM call error: {e}", file=sys.stderr)
            return ""

        # The parsed review is the only place the per-dimension ratings exist:
        # this method returns just the decision word, so without a span the
        # Soundness / Significance / Overall the rubric produced are discarded
        # and the architect has no way to see WHY a paper was rejected.
        with _span("extract_review", "function") as set_attr:
            set_attr("ce.inputs.response_chars", len(content or ""))
            review = _extract_last_json(content or "")
            if not review or "Decision" not in review:
                msg = "could not extract Decision from response"
                set_attr("ce.error", msg)
                set_attr("ce.output", (content or "")[:2000])
                print(f"[review] {msg}", file=sys.stderr)
                return ""
            scores = {k: review.get(k) for k in REVIEW_SCORE_FIELDS if k in review}
            set_attr("ce.output", json.dumps(scores))
            # Also on stderr: `eval_stderr.log` keeps this even for an eval whose
            # spans are never opened, and it is the aggregate view across rows.
            print(f"[review] basis={json.dumps(scores)}", file=sys.stderr)
        return str(review["Decision"]).strip().lower()
