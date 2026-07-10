# Metric: binary accuracy of the Accept/Reject decision, with textual feedback.
#
# Contract (mounted evaluator): called as fn(output=..., example=...)
# where `example` is an attribute-access row object (NOT a dict). Returns
# {"score": float, "feedback": str}. Unparseable/empty predictions score 0.0 —
# they are NOT dropped from the denominator (stricter than the original
# benchmark's report.py, which filtered them out).

VALID = {"accept", "reject"}


def _normalize(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("decision") or value.get("Decision") or ""
    return str(value).strip().lower()


def review_decision_accuracy(*args, **kwargs) -> dict:
    output = kwargs.get("output", args[0] if args else None)

    gold = None
    question_id = "unknown"
    example = kwargs.get("example")
    if example is not None:
        gold = _normalize(getattr(example, "outcome", None))
        question_id = getattr(example, "question_id", "unknown")
    if not gold:
        return {
            "score": 0.0,
            "feedback": "No ground-truth outcome available to score against.",
        }

    pred = _normalize(output)

    if pred not in VALID:
        return {
            "score": 0.0,
            "feedback": (
                f"Paper {question_id}: prediction {pred!r} is not a valid "
                f"decision (expected 'accept' or 'reject'). The true outcome "
                f"was '{gold}'. Empty or malformed predictions score 0 — the "
                f"program must always return a parseable decision."
            ),
        }

    if pred == gold:
        return {
            "score": 1.0,
            "feedback": f"Paper {question_id}: correct — predicted '{pred}' and the venue decision was '{gold}'.",
        }
    return {
        "score": 0.0,
        "feedback": (
            f"Paper {question_id}: wrong — predicted '{pred}' but the venue "
            f"decision was '{gold}'. Consider what signals in the paper "
            f"(soundness of evaluation, novelty, clarity, significance) the "
            f"review process weighed differently than your rubric did."
        ),
    }
