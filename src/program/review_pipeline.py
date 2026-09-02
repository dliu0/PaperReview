# Seed program: LLM paper reviewer producing an Accept/Reject decision.
# Prompt adapted from the meta-hyperagents paper_review domain (itself adapted
# from SakanaAI/AI-Scientist perform_review.py).

import json
import re
import sys

# Provider wiring lives OUTSIDE this package, in `src/lm_provider.py`: the
# pinned model, the endpoints, the keys, the streaming hang guard and the
# cross-provider fallback are benchmark infrastructure, not part of the review
# procedure being optimized (docs/provider_fallback.md, C8). `$LM_PROVIDER` /
# `$LM_FALLBACK` repoint the provider, name the cover, or disarm the divert
# without editing code. Nothing below names a model, an endpoint or a key.
from src.lm_provider import REASONING_EFFORT, CallStats, build_task_lm

# Tracing lives in `_tracing.py` (the CodeEvolver skill's `traceable.py` plus
# two local helpers). Nothing auto-instruments a raw `litellm.completion`, so
# without these spans this pipeline produces exactly one span carrying its
# final output -- the single word "accept"/"reject" `__call__` returns -- and
# the review the model actually wrote (the THOUGHT, the per-dimension ratings,
# the Overall score) is computed and thrown away unseen. The architect cannot
# improve a rubric whose output it cannot read.
from ._tracing import span, summarize_messages, traceable

# The full response is the point of the `llm` span; this only guards against a
# runaway generation, not against ordinary long reviews.
RESPONSE_MAX_CHARS = 20000


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
        # The solver LM: the pinned model on whichever provider is currently
        # serving it, with the next provider down the preference order covering
        # for it per call. Building it here (rather than at call time) means a
        # missing key or an unroutable provider fails before the eval starts.
        self.lm = build_task_lm()

    def _complete(self, messages):
        """One routed completion; returns the response text.

        Emits one `llm` span per call recording the request, the FULL response
        text, which provider actually served it, and how the hang guard behaved
        (attempts, stream chunks, elapsed, the error of every failed attempt).
        Retries and provider diverts are the failure modes this program is most
        exposed to, and they were previously invisible.

        `src/lm_provider.py` also stamps `lm.fallback.*` / `lm.breaker.*` /
        `lm.primary_skipped` on this same span when a call is diverted, so a
        row served by the cover says so in the trace.
        """
        stats = CallStats()
        failure = None
        with span("review_llm", "llm") as set_attr:
            set_attr("ce.inputs.messages", summarize_messages(messages))
            set_attr("ce.inputs.model", self.lm.model)
            if self.lm.api_base:
                set_attr("ce.inputs.api_base", self.lm.api_base)
            set_attr("ce.inputs.reasoning_effort", REASONING_EFFORT)
            set_attr("gen_ai.request.model", self.lm.model)
            try:
                content = self.lm.complete(messages, stats=stats)
            except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised
                failure = exc
            else:
                set_attr("ce.output", content[:RESPONSE_MAX_CHARS])
                set_attr("response_chars", len(content))

            set_attr("attempts", stats.attempts)
            set_attr("stream_chunks", stats.stream_chunks)
            set_attr("elapsed_s", stats.elapsed_s)
            set_attr("lm.provider", stats.provider)
            set_attr("lm.served_by", stats.served_by)

            if failure is None:
                if stats.errors:
                    set_attr("recovered_after_errors", json.dumps(stats.errors))
                return content
            set_attr(
                "ce.error",
                json.dumps(stats.errors) if stats.errors else "completion failed",
            )
        # Outside the span: a failed call is this program's data, not a broken
        # span, and marking the span ERROR would hide `ce.error` behind a
        # recorded exception.
        raise failure

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

        try:
            scores = self._extract_review(content or "")
        except ValueError as e:
            print(f"[review] {e}", file=sys.stderr)
            return ""
        # Also on stderr: `eval_stderr.log` keeps this even for an eval whose
        # spans are never opened, and it is the aggregate view across rows.
        print(f"[review] basis={json.dumps(scores)}", file=sys.stderr)
        return str(scores["Decision"]).strip().lower()

    # `max_attr_chars` bounds the response echoed back as this span's input;
    # the full text is already on the `llm` span above. The ratings it returns
    # exist nowhere else -- `__call__` returns only the decision word -- so
    # without this span the architect cannot see WHY a paper was rejected.
    @traceable("function", name="extract_review", max_attr_chars=2000)
    def _extract_review(self, content: str) -> dict:
        """Per-dimension ratings from the response, or `ValueError` if unusable.

        Raising keeps "the provider failed" distinguishable from "the rubric
        produced garbage": the first leaves `ce.error` on `review_llm`, the
        second leaves it here, on a span whose input is the unusable text.
        Malformed decisions score 0 and stay in the denominator, so which one
        happened matters.
        """
        review = _extract_last_json(content)
        if not review or "Decision" not in review:
            raise ValueError("could not extract Decision from response")
        return {k: review.get(k) for k in REVIEW_SCORE_FIELDS if k in review}
