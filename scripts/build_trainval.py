#!/usr/bin/env python3
"""Build data_splits/trainval.json — the combined train+val training pool.

Engine 3 removed the validation set entirely: `valset_path` / `max_valset_size`
are rejected by `OptimizationConfig`, and the architect compares branches on a
fixed comparison set drawn from the TRAINSET. So the 100/100 train/val split is
folded back into one 200-row training pool that every arm (asa, greedy,
pareto_evolution, vanilla) trains on identically — which is also the equal-data
control the vanilla baseline needed under engine 2.

The engine refuses to merge splits by design (`DatasetConfig`), so building the
combined file is the caller's job — this script. Committed rather than left as a
working-tree file: the seed bundle is `git archive` of one branch, so anything
untracked simply does not reach the runner image (and an earlier untracked copy
of this pair was lost exactly that way).

Output is an exact ORDERED concatenation of trainset.json (100) then
valset.json (100) = 200 rows, NOT deduplicated — dropping a row that appears in
both splits would change the training pool relative to what the splits describe.
Disjointness from the blind test hold-out is asserted on question_id.

    python scripts/build_trainval.py
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(HERE, "data_splits", "trainset.json")
VAL = os.path.join(HERE, "data_splits", "valset.json")
TEST = os.path.join(HERE, "tests", "testdata", "testset.json")
OUT = os.path.join(HERE, "data_splits", "trainval.json")
ID_KEY = "question_id"


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    train, val, test = _load(TRAIN), _load(VAL), _load(TEST)
    combined = list(train) + list(val)  # ordered concat, no dedup

    test_ids = {r[ID_KEY] for r in test}
    leaked = sorted({r[ID_KEY] for r in combined} & test_ids)
    assert not leaked, f"train+val leaks into test hold-out: {leaked[:10]}"

    tv_ids = [r[ID_KEY] for r in combined]
    internal_dupes = len(tv_ids) - len(set(tv_ids))

    with open(OUT, "w") as f:
        json.dump(combined, f)
    print(
        f"wrote {OUT}: {len(combined)} rows "
        f"= train({len(train)}) + val({len(val)}); "
        f"{internal_dupes} train/val overlap rows kept; disjoint from test({len(test)})"
    )


if __name__ == "__main__":
    main()
