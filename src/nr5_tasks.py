"""V5 task wrappers: unchanged task semantics and target-blind prompt interventions.

All canonical training/evaluation prompts delegate to the archived V2 renderer.
The intervention prompts change demonstrations only. Privileged correct prefixes
are diagnostic labels kept separately from target-blind query examples.
"""
from __future__ import annotations
import hashlib
import json
import random
import re
from typing import Any
import rlq_tasks as core

TASK_NAMES = core.TASK_NAMES
OPERATIONS = core.OPERATIONS
solve = core.solve
extract_answer = core.extract_answer
extract_trace = core.extract_trace
format_prompt = core.format_prompt
DEMOS = core.DEMOS
VERSION = "native-routing-v5-depth-interventions-20261006"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


THREE_DEMOS = {
    0: core._example(0, 3, core._arithmetic_problem(2, [("add", 5), ("multiply", 3), ("subtract", 4)])),
    1: core._example(1, 3, core._pointer_problem(6, core._cycle([0, 2, 5, 8, 1, 4, 7, 9, 3, 6]), 3)),
}
ALL_DEMOS = [e for rows in DEMOS.values() for e in rows] + list(THREE_DEMOS.values())
DEMO_IDS = {e["id"] for e in ALL_DEMOS}
VARIANTS = ("standard", "reverse", "one_only", "three_only")


def variant_prompt(example: dict[str, Any], variant: str = "standard") -> str:
    """Target-blind diagnostic renderer; standard is byte-identical to V2.

Reverse changes only demo order. One_only and three_only change demonstration
number and length, so they are contextual interventions, not matched ablations.
    """
    if variant == "standard":
        return format_prompt(example)
    task_id = int(example["task_id"])
    if variant == "reverse":
        demos = list(reversed(DEMOS[task_id]))
    elif variant == "one_only":
        demos = DEMOS[task_id][:1]
    elif variant == "three_only":
        demos = [THREE_DEMOS[task_id]]
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")
    parts = ["Solve the problem. Show the starting digit and each next digit. End with a line: Answer: d"]
    for demo in demos:
        path = " -> ".join(map(str, demo["gold_trace"]))
        parts.append(f"Problem: {demo['problem']}\nWork: {path}\nAnswer: {demo['answer']}")
    parts.append(f"Problem: {example['problem']}\nWork:")
    return "\n\n".join(parts)


def pointer_parts(example: dict[str, Any]) -> tuple[int, dict[int, int]]:
    if int(example["task_id"]) != 1:
        raise ValueError("Expected pointer example")
    problem = example["problem"]
    match = re.search(r"Start at ([0-9])\.", problem)
    mapping = {int(a): int(b) for a, b in re.findall(r"([0-9])->([0-9])", problem)}
    if match is None or len(mapping) != 10 or set(mapping.values()) != set(range(10)):
        raise ValueError("Malformed pointer example")
    return int(match.group(1)), mapping


def pointer_map_key(example: dict[str, Any]) -> str:
    _, mapping = pointer_parts(example)
    return _digest([mapping[i] for i in range(10)])[:24]


def pointer_family_key(example: dict[str, Any]) -> str:
    start, mapping = pointer_parts(example)
    return _digest({"start": start, "map": [mapping[i] for i in range(10)]})[:24]


def with_pointer_depth(example: dict[str, Any], depth: int, include_labels: bool = True) -> dict[str, Any]:
    """Construct a counterfactual request from old or new public problem text."""
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("Depth must be a positive integer")
    start, mapping = pointer_parts(example)
    row = core._example(1, depth, core._pointer_problem(start, mapping, depth))
    row["family_id"] = pointer_family_key(row)
    if not include_labels:
        row = {k: v for k, v in row.items() if k not in ("answer", "gold_trace")}
    return row


def build_pointer_counterfactuals(seed: int = 20261006, nfamilies: int = 20,
                                  depths=(1, 2, 3, 4), excluded_families=(),
                                  excluded_maps=()) -> dict[str, Any]:
    """Paired public queries plus separate solver labels, never model-selected.

One start is used per fresh map. Starts cycle through independently shuffled
permutations of0..9; answer digits are not balanced. Query depth changes within
each map/start family. Maps are ten-cycles, and therefore depth two and three
have different correct answers. These are paired descriptive diagnostics.
The two-hop prefix is privileged assistance, including when depth<2; those
overrun contexts are useful stopping diagnostics but not natural task accuracy.
    """
    depths = tuple(int(d) for d in depths)
    if not isinstance(nfamilies, int) or nfamilies <= 0 or not depths or min(depths) < 1 or len(set(depths)) != len(depths):
        raise ValueError("nfamilies must be a positive integer; depths unique and positive")
    excluded_families = set(excluded_families)
    excluded_maps = set(excluded_maps) | {pointer_map_key(e) for e in ALL_DEMOS if e["task_id"] == 1}
    rng = random.Random(core._seed(VERSION, "counterfactual", seed))
    queries, labels, chosen_maps = [], {}, set()
    starts = []
    while len(starts) < nfamilies:
        digits = list(range(10)); rng.shuffle(digits)
        starts.extend(digits)
    starts = starts[:nfamilies]
    attempts = 0
    while len(chosen_maps) < nfamilies:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("Could not allocate fresh diagnostic maps")
        order = list(range(10)); rng.shuffle(order)
        mapping = core._cycle(order)
        prototype = core._example(1, 1, core._pointer_problem(0, mapping, 1))
        map_key = pointer_map_key(prototype)
        if map_key in excluded_maps | chosen_maps:
            continue
        start = starts[len(chosen_maps)]
        family = core._example(1, 1, core._pointer_problem(start, mapping, 1))
        if pointer_family_key(family) in excluded_families:
            continue
        chosen_maps.add(map_key)
        three = core.solve(1, core._pointer_problem(start, mapping, 3))["gold_trace"]
        prefix = " " + " -> ".join(map(str, three[:3]))
        for depth in depths:
            row = with_pointer_depth(family, depth)
            labels[row["id"]] = {"answer": row.pop("answer"), "gold_trace": row.pop("gold_trace"),
                                  "gold_prefix_two": prefix,
                                  "gold_next_digit_after_two": str(three[3])}
            queries.append(row)
    queries.sort(key=lambda e: (e["family_id"], e["difficulty"]))
    return {"examples": queries, "labels": labels,
            "provenance": {"version": VERSION, "seed": seed, "n_families": nfamilies,
                           "n_maps": len(chosen_maps), "depths": list(depths),
                           "start_counts": {str(d): starts.count(d) for d in range(10)},
                           "start_balance": "Shuffled cycles of0..9; exact if n_families is a multiple of10.",
                           "answer_balance": "Not constrained; descriptive paired diagnostic, not a balanced accuracy benchmark.",
                           "map_keys": sorted(chosen_maps), "examples_sha256": _digest(queries),
                           "labels_sha256": _digest(labels),
                           "scope": "Privileged-prefix and prompt interventions are diagnostics, not unassisted accuracy."}}
