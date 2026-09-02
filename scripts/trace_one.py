#!/usr/bin/env python3
"""Run ReviewPipeline on one real row and dump its spans to traces/traces.jsonl.

`tests/test_tracing.py` already pins the span shape, but it does so against a
stubbed litellm: it proves the attributes are set, not that they survive a real
run. This script closes the three gaps the tests structurally cannot reach.

  1. A real streamed completion -- `stream_chunks`, `elapsed_s` and the hang
     guard's retry path against the actual provider, not a list of fake chunks.
  2. A real ~50k-char paper, so the payload bound is measured on the input it
     was written for rather than on `"PAPER BODY " * 5000`.
  3. The lazy tracer resolving against a provider installed AFTER
     `review_pipeline` was imported -- the ordering the engine uses, and the
     one way a span can silently go to a no-op tracer.

Outside CodeEvolver, OTel installs a no-op provider and the spans evaporate, so
this installs the skill's local file exporter first. Inside the orchestrator
that exporter is redundant (the harness writes `/traces/iteration_{N}_{suffix}/`)
and nothing in `src/program/` imports it.

The harness wraps each row in a root span carrying the program's output. Nothing
does that locally, and without it the two spans are separate roots and land on
separate lines. This opens an equivalent root so the dump has the same shape the
architect actually reads.

    pip install opentelemetry-sdk        # not in requirements.txt; see below
    export GMI_API_KEY=...               # or GMI_CLOUD_API_KEY
    export DEEPINFRA_API_KEY=...         # the cover; see docs/provider_fallback.md
    python scripts/trace_one.py                     # row 0 of the trainset
    python scripts/trace_one.py --row 3 --split data_splits/valset.json
    LM_PROVIDER=deepinfra LM_FALLBACK=0 python scripts/trace_one.py   # pin one provider

Costs one real DeepSeek-V4-Flash call at reasoning_effort=high -- minutes, not
seconds. `--dry-run` exercises the whole path with a stub instead.
"""
import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

DEFAULT_SPLIT = os.path.join("data_splits", "trainset.json")
DEFAULT_OUT = os.path.join("traces", "traces.jsonl")

STUB_RESPONSE = (
    "THOUGHT:\nStubbed response -- no provider was called.\n\nREVIEW JSON:\n"
    "```json\n"
    + json.dumps(
        {
            "Summary": "a stubbed summary",
            "Originality": 2,
            "Soundness": 3,
            "Significance": 2,
            "Overall": 5,
            "Confidence": 4,
            "Decision": "Reject",
        }
    )
    + "\n```"
)


def _install_stub_litellm():
    """Stand in for the provider so `--dry-run` still streams and still spans."""

    def completion(**kw):
        chunk = lambda c: types.SimpleNamespace(  # noqa: E731
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=c))]
        )
        return [chunk(STUB_RESPONSE[i : i + 40]) for i in range(0, len(STUB_RESPONSE), 40)]

    fake = types.ModuleType("litellm")
    fake.completion = completion
    sys.modules["litellm"] = fake


def _summarize(out_path):
    """Print one line per span: the tree, the kinds, and what each cost."""
    with open(out_path) as f:
        bundles = [json.loads(line) for line in f if line.strip()]

    for bundle in bundles:
        print(f"\ntrace {bundle['trace_id'][:16]}  root={bundle['root_name']}  "
              f"{bundle['duration_ms']:.0f}ms")
        by_parent = {}
        for sp in bundle["spans"]:
            by_parent.setdefault(sp["parent_id"], []).append(sp)

        def walk(parent_id, depth):
            for sp in by_parent.get(parent_id, []):
                attrs = sp["attributes"]
                ms = (sp["end_time_ns"] - sp["start_time_ns"]) / 1e6
                kind = attrs.get("ce.span_kind", "?")
                print(f"{'  ' * depth}- {sp['name']} [{kind}] {ms:.0f}ms "
                      f"status={sp['status']}")
                for key in sorted(attrs):
                    if not key.startswith("ce.") or key == "ce.span_kind":
                        continue
                    value = str(attrs[key])
                    head = value.replace("\n", " ")[:100]
                    print(f"{'  ' * depth}    {key} ({len(value)} chars): {head}")
                # Non-`ce.*` attributes land in `metadata` on the trace entry.
                # The hang guard's telemetry lives here -- attempts, chunks and
                # elapsed are the whole point of the `llm` span, so show them.
                meta = {k: v for k, v in sorted(attrs.items())
                        if not k.startswith("ce.")}
                if meta:
                    print(f"{'  ' * depth}    metadata: {meta}")
                walk(sp["span_id"], depth + 1)

        walk(None, 1)
        payload = sum(len(str(k)) + len(str(v))
                      for sp in bundle["spans"] for k, v in sp["attributes"].items())
        print(f"  total span payload: {payload} chars")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--row", type=int, default=0, help="index into the split")
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="stub the provider: exercises tracing without an LLM call")
    args = ap.parse_args()

    if args.dry_run:
        _install_stub_litellm()
        # Routing still runs for real, and the GMI route reads its key at
        # construction; a dry run has no reason to demand a real one.
        os.environ.setdefault("GMI_API_KEY", "dry-run")

    # Which provider serves this, and which one covers for it, is decided by
    # src/lm_provider.py ($LM_PROVIDER / $LM_FALLBACK) -- not by this script.
    # Build the LM before touching the dataset so a missing key fails in one
    # line rather than after a minutes-long call.
    from src.lm_provider import build_task_lm

    try:
        lm = build_task_lm()
    except KeyError as exc:
        sys.exit(f"provider key missing: {exc.args[0]}; or pass --dry-run")
    print(f"provider: {lm.provider} ({lm.model})"
          + (f" -> {lm.fallback_provider} ({lm.fallback_model})"
             if lm.fallback_provider else "  [fallback disarmed]"))

    with open(os.path.join(HERE, args.split)) as f:
        rows = json.load(f)
    if not 0 <= args.row < len(rows):
        sys.exit(f"--row {args.row} out of range for {args.split} ({len(rows)} rows)")
    row = rows[args.row]

    # Import the program BEFORE installing the provider: this is the ordering
    # the engine uses, and the reason the tracer is fetched lazily.
    from src.metric.metric import review_decision_accuracy
    from src.program._tracing import span
    from src.program._tracing_export import install_file_exporter
    from src.program.review_pipeline import ReviewPipeline

    out_path = install_file_exporter(os.path.join(HERE, args.out))

    print(f"row {args.row}: {row['question_id']}  "
          f"paper={len(row['paper_text'])} chars  gold={row['outcome']}")

    with span("paper_review_row", "chain") as set_attr:  # stands in for the harness root
        set_attr("ce.inputs.question_id", row["question_id"])
        decision = ReviewPipeline()(paper_text=row["paper_text"])
        set_attr("ce.output", decision)

    result = review_decision_accuracy(
        output=decision, example=types.SimpleNamespace(**row)
    )
    print(f"\ndecision={decision!r}  score={result['score']}\n{result['feedback']}")

    _summarize(out_path)
    print(f"\nfull dump: {out_path}")


if __name__ == "__main__":
    main()
