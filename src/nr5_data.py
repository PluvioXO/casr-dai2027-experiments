"""V5 fixed-depth versus mixed-depth data with common held-out evaluation.

No answers enter prompts. Every training/development/test stratum is balanced
across final digits; separate paired diagnostics need not have balanced answers.
Pointer maps (not merely exact questions) are disjoint across splits; within a
split the same maps and starts occur at every selected depth. Arithmetic holds
out exact problems, not operation primitives or program prefixes.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import rlq_tasks as core
import nr5_tasks as tasks

VERSION = "native-routing-v5-crossed-depth-data-20261006"
DATA_SEED = 20261006
TASK_CODE_SHA256 = "aee20876e0c7d7811cd989311ef3f0d4203347c2fe84387e62e1e7a2e28de317"
SPLITS = ("train", "dev", "test", "depth_probe")
PROFILES = {
    "full": {"train": 240, "train_a1": 80, "dev": 40, "dev_a1": 20, "test": 60, "depth_probe": 60, "diagnostic_families": 20},
    "pilot": {"train": 120, "train_a1": 80, "dev": 40, "dev_a1": 20, "test": 60, "depth_probe": 60, "diagnostic_families": 20},
    "smoke": {"train": 10, "train_a1": 10, "dev": 10, "dev_a1": 10, "test": 10, "depth_probe": 10, "diagnostic_families": 10},
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def problem_key(example):
    return int(example["task_id"]), example["problem"]


def _levels(split):
    return (4, 5) if split == "depth_probe" else (1, 2, 3)


def _count(profile, split, task_id, depth):
    key = f"{split}_a1" if task_id == 0 and depth == 1 and split in ("train", "dev") else split
    return PROFILES[profile][key]


def _take_balanced(rows, count):
    selected = []
    for digit in map(str, range(10)):
        candidates = sorted((e for e in rows if e["answer"] == digit), key=lambda e: e["id"])
        if len(candidates) < count // 10:
            raise AssertionError("Insufficient digit-balanced stratum")
        selected.extend(candidates[:count // 10])
    return selected


def _full_pools():
    """Generate full-sized reserves before regime/profile filtering."""
    pools = {s: {} for s in SPLITS}
    used = set(tasks.DEMO_IDS)
    # The finite A1 domain has210 problems; one canonical A1demo is excluded.
    # Reserve8/2/6 examples per digit for train/dev/test, well inside capacity.
    for answer in range(10):
        rows = [core._example(0, 1, core._arithmetic_problem(core._reverse(answer, op), [op])) for op in core.OPERATIONS]
        rows = [e for e in rows if e["id"] not in used]
        rows.sort(key=lambda e: core._digest(f"{VERSION}|a1|{e['id']}"))
        offset = 0
        for split, per_digit in (("train", 8), ("dev", 2), ("test", 6)):
            chosen = rows[offset:offset + per_digit]
            assert len(chosen) == per_digit
            pools[split].setdefault((0, 1), []).extend(chosen)
            used.update(e["id"] for e in chosen)
            offset += per_digit
    for split in SPLITS:
        for depth in _levels(split):
            if depth == 1:
                continue
            rows = []
            count = _count("full", split, 0, depth)
            for answer in range(10):
                rng = random.Random(core._seed(VERSION, DATA_SEED, split, "arithmetic", depth, answer))
                added = 0
                while added < count // 10:
                    e = core._candidate(rng, 0, depth, answer)
                    if e["id"] in used:
                        continue
                    used.add(e["id"]); rows.append(e); added += 1
            pools[split][(0, depth)] = rows
    used_maps = {tasks.pointer_map_key(e) for e in tasks.ALL_DEMOS if e["task_id"] == 1}
    pointer_maps = {}
    for split in SPLITS:
        rng = random.Random(core._seed(VERSION, DATA_SEED, split, "pointer_maps"))
        mappings = []
        count = PROFILES["full"][split]
        while len(mappings) < count // 10:
            digits = list(range(10)); rng.shuffle(digits)
            mapping = core._cycle(digits)
            example = core._example(1, 1, core._pointer_problem(0, mapping, 1))
            key = tasks.pointer_map_key(example)
            if key in used_maps:
                continue
            used_maps.add(key); mappings.append(mapping)
        pointer_maps[split] = mappings
        for depth in _levels(split):
            rows = []
            for mapping in mappings:
                for start in range(10):
                    row = core._example(1, depth, core._pointer_problem(start, mapping, depth))
                    row["family_id"] = tasks.pointer_family_key(row)
                    assert row["id"] not in used
                    used.add(row["id"]); rows.append(row)
            pools[split][(1, depth)] = rows
    return pools, used_maps


def audit(data):
    """Recompute labels, balanced counts, exact and map-level separation."""
    regime = data["provenance"]["regime"]
    profile = data["provenance"]["profile"]
    seen_ids = set(tasks.DEMO_IDS)
    seen_keys = {problem_key(e) for e in tasks.ALL_DEMOS}
    demo_maps = {tasks.pointer_map_key(e) for e in tasks.ALL_DEMOS if e["task_id"] == 1}
    map_owner = {key: "demo" for key in demo_maps}
    counts = {}
    for split in SPLITS:
        strata = {}
        for e in data[split]:
            assert e["id"] not in seen_ids and problem_key(e) not in seen_keys, f"Content leak in {split}"
            seen_ids.add(e["id"]); seen_keys.add(problem_key(e))
            solved = core.solve(e["task_id"], e["problem"])
            assert solved == {"answer": e["answer"], "gold_trace": e["gold_trace"]}
            assert len(e["gold_trace"]) == e["difficulty"] + 1
            assert core._example(e["task_id"], e["difficulty"], e["problem"])["id"] == e["id"]
            poison = dict(e, answer="TARGET_LEAK", gold_trace=["TARGET_LEAK"])
            for variant in tasks.VARIANTS:
                assert tasks.variant_prompt(e, variant) == tasks.variant_prompt(poison, variant)
            strata.setdefault((e["task_id"], e["difficulty"]), Counter())[e["answer"]] += 1
            if e["task_id"] == 1:
                mk = tasks.pointer_map_key(e)
                assert map_owner.get(mk, split) == split, f"Pointer map leaks into {split}"
                map_owner[mk] = split
                assert e["family_id"] == tasks.pointer_family_key(e)
        expected = {(t, d): _count(profile, split, t, d) for t in (0, 1) for d in _levels(split)
                    if split != "train" or regime == "mixed" or (t, d) in ((0, 1), (1, 2))}
        assert set(strata) == set(expected)
        for key, n in expected.items():
            assert strata[key] == Counter({str(d): n // 10 for d in range(10)}), (split, key)
        counts[split] = {f"task_{t}_depth_{d}": n for (t, d), n in expected.items()}
    family_depths = {}
    diagnostic_map_families = {}
    for e in data["diagnostic"]:
        assert "answer" not in e and "gold_trace" not in e
        assert e["id"] not in seen_ids and problem_key(e) not in seen_keys
        seen_ids.add(e["id"]); seen_keys.add(problem_key(e))
        mk = tasks.pointer_map_key(e)
        assert map_owner.get(mk, "diagnostic") == "diagnostic"
        map_owner[mk] = "diagnostic"
        label = data["diagnostic_labels"][e["id"]]
        assert core.solve(1, e["problem"]) == {"answer": label["answer"], "gold_trace": label["gold_trace"]}
        family_depths.setdefault(e["family_id"], set()).add(e["difficulty"])
        diagnostic_map_families.setdefault(mk, set()).add(e["family_id"])
    assert all(depths == {1, 2, 3, 4} for depths in family_depths.values())
    assert len(diagnostic_map_families) == len(family_depths)
    assert all(len(families) == 1 for families in diagnostic_map_families.values())
    return {"passed": True, "counts": counts, "diagnostic_families": len(family_depths),
            "checks": ["exact ID and content separation", "solver labels", "digit balance", "demo exclusion",
                       "pointer maps disjoint across splits", "target-blind prompt variants", "paired diagnostic depths",
                       "one distinct map per diagnostic family"]}


def make_data(regime="fixed", profile="pilot"):
    if regime not in ("fixed", "mixed") or profile not in PROFILES:
        raise ValueError("Unknown regime/profile")
    actual_task_hash = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    if actual_task_hash != TASK_CODE_SHA256:
        raise AssertionError("Frozen rlq_tasks.py changed; do not mix task semantics")
    pools, reserved_maps = _full_pools()
    data = {s: [] for s in SPLITS}
    for split in SPLITS:
        for (task_id, depth), rows in pools[split].items():
            if split == "train" and regime == "fixed" and (task_id, depth) not in ((0, 1), (1, 2)):
                continue
            count = _count(profile, split, task_id, depth)
            if task_id == 1:
                # Keep complete maps/all10starts. A digit-stratified row subsample
                # would destroy matched map/start families across requested depth.
                selected = rows[:count]
            else:
                selected = _take_balanced(rows, count)
            data[split].extend(selected)
        data[split].sort(key=lambda e: e["id"])
    diagnostic = tasks.build_pointer_counterfactuals(seed=DATA_SEED, nfamilies=PROFILES[profile]["diagnostic_families"],
                                                    excluded_maps=reserved_maps)
    data["diagnostic"] = diagnostic["examples"]
    data["diagnostic_labels"] = diagnostic["labels"]
    data["provenance"] = {
        "version": VERSION, "regime": regime, "profile": profile, "data_seed": DATA_SEED,
        "task_code_sha256": actual_task_hash,
        "split_sha256": {s: json_hash(data[s]) for s in SPLITS},
        "diagnostic_sha256": json_hash(data["diagnostic"]), "diagnostic_labels_sha256": json_hash(data["diagnostic_labels"]),
        "diagnostic": diagnostic["provenance"],
        "standard_demonstration_depths": [1, 2], "prompt_variants_used_in_training": ["standard"],
        "evaluation_pairing": "Dev/test/depth_probe identical across fixed/mixed and pilot/full; smoke is a smaller software check.",
        "selection": "Select checkpoints on the common mixed-depth dev bank only. Fixed training also uses this mixed-depth selection, so it is not an exact V4 replication. Test/depth_probe/diagnostics never select checkpoints.",
        "scope": "Exploratory: fresh deterministic split, but finite one-step arithmetic can overlap prior historical evaluations. Not a new untouched confirmatory task domain.",
        "caveat": "Arithmetic holds out exact problems only; primitives and program prefixes may recur. Pointer maps are disjoint by split; constituent edges may recur.",
        "depth_sampling": "Runner should sample tasks and then available depths uniformly; fixed/mixed comparisons hold update count, not number of distinct training problems, constant.",
        "finite_a1_domain": {"total": 210, "train": 80, "dev": 20, "test": 60, "excluded_a1_demos": 1},
    }
    checked = audit(data)
    data["provenance"]["counts"] = checked["counts"]
    data["provenance"]["audit"] = checked
    return data


def self_test():
    fixed = make_data("fixed", "pilot")
    mixed = make_data("mixed", "pilot")
    full = make_data("mixed", "full")
    smoke = make_data("mixed", "smoke")
    assert fixed == make_data("fixed", "pilot")
    for split in ("dev", "test", "depth_probe", "diagnostic", "diagnostic_labels"):
        assert fixed[split] == mixed[split] == full[split]
    for split in ("train", "dev"):
        assert {e["id"] for e in fixed[split]} <= {e["id"] for e in mixed[split]}
    # Pointer family/depth pairing is preserved even when the pilot is smaller.
    for dataset in (mixed, full, smoke):
        for split in ("train", "dev", "test", "depth_probe"):
            groups = {}
            for e in dataset[split]:
                if e["task_id"] == 1:
                    groups.setdefault(e["family_id"], set()).add(e["difficulty"])
            assert all(depths == set(_levels(split)) for depths in groups.values())
    return {"passed": True, "fixed": fixed["provenance"]["audit"],
            "mixed": mixed["provenance"]["audit"], "full_counts": full["provenance"]["counts"],
            "checks": ["repeatability", "common evaluation across regimes/profiles", "fixed training subset of mixed", "pointer depth counterfactual pairing"]}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
