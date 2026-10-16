"""Balanced, leakage-checked tasks for the routed-RL learning qualification notebook.

Standard library only. States and answers are digits. Arithmetic uses invertible
non-identity operations modulo 10; pointer maps are single cycles of ten digits.
The scalar training reward may use ``answer`` but prompts never read target
answers or traces. All held-out sets are fixed across adaptation seeds.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re
from typing import Any

TASK_NAMES = {0: "arithmetic", 1: "pointer_following"}
SPLIT_VERSION = "rlq-v2-digit-content-split-20261004"
EVALUATION_SEED = 20261004
OPERATIONS = tuple((kind, n) for kind in ("add", "subtract") for n in range(1, 10)) + tuple(("multiply", n) for n in (3, 7, 9))
ONE_STEP_CAPS = {"train": 80, "dev": 20, "test": 20}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed(*parts: Any) -> int:
    return int(_digest("|".join(map(str, parts)))[:16], 16)


def _apply(state: int, operation: tuple[str, int]) -> int:
    kind, value = operation
    if kind == "add":
        return (state + value) % 10
    if kind == "subtract":
        return (state - value) % 10
    if kind == "multiply":
        return (state * value) % 10
    raise ValueError(f"Unknown operation: {kind}")


def _reverse(state: int, operation: tuple[str, int]) -> int:
    kind, value = operation
    if kind == "add":
        return (state - value) % 10
    if kind == "subtract":
        return (state + value) % 10
    if kind == "multiply":
        return (state * pow(value, -1, 10)) % 10
    raise ValueError(f"Unknown operation: {kind}")


def _arithmetic_problem(start: int, operations: list[tuple[str, int]]) -> str:
    instructions = "; ".join(f"{kind} by {n}" if kind == "multiply" else f"{kind} {n}" for kind, n in operations)
    return (f"Start at {start}. After each operation, take the remainder modulo 10, always in 0–9.\n"
            f"Operations: {instructions}.\nWhat is the final digit?")


def _pointer_problem(start: int, mapping: dict[int, int], hops: int) -> str:
    map_text = ", ".join(f"{n}->{mapping[n]}" for n in range(10))
    return (f"Follow this map: {map_text}.\nStart at {start}. Follow exactly {hops} links.\n"
            "What is the final digit?")


def solve(task_id: int, problem: str) -> dict[str, Any]:
    """Solve the public problem text, returning a digit string and integer path."""
    start_match = re.search(r"Start at ([0-9])\.", problem)
    if start_match is None:
        raise ValueError("Problem has no valid start digit")
    trace = [int(start_match.group(1))]
    if task_id == 0:
        operations_match = re.search(r"Operations: (.+)\.\n", problem)
        if operations_match is None:
            raise ValueError("Arithmetic problem has no operation list")
        for instruction in operations_match.group(1).split("; "):
            match = re.fullmatch(r"(add|subtract|multiply)(?: by)? ([1-9])", instruction)
            if match is None:
                raise ValueError(f"Invalid operation: {instruction}")
            operation = (match.group(1), int(match.group(2)))
            if operation not in OPERATIONS:
                raise ValueError(f"Operation outside task domain: {operation}")
            trace.append(_apply(trace[-1], operation))
    elif task_id == 1:
        mapping = {int(a): int(b) for a, b in re.findall(r"([0-9])->([0-9])", problem)}
        hops_match = re.search(r"Follow exactly ([1-9][0-9]*) links\.", problem)
        if len(mapping) != 10 or set(mapping.values()) != set(range(10)) or hops_match is None:
            raise ValueError("Invalid pointer problem")
        for _ in range(int(hops_match.group(1))):
            trace.append(mapping[trace[-1]])
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
    return {"answer": str(trace[-1]), "gold_trace": trace}


def _example(task_id: int, difficulty: int, problem: str) -> dict[str, Any]:
    example = {
        "id": _digest(f"{SPLIT_VERSION}|{task_id}|{problem}")[:24],
        "task_id": task_id,
        "difficulty": difficulty,
        "problem": problem,
    }
    example.update(solve(task_id, problem))
    if len(example["gold_trace"]) != difficulty + 1:
        raise AssertionError("Difficulty/path mismatch")
    return example


def _cycle(values: list[int]) -> dict[int, int]:
    return {values[i]: values[(i + 1) % len(values)] for i in range(len(values))}


# Fixed prompt demonstrations are excluded from every generated split.
DEMOS = {
    0: [
        _example(0, 1, _arithmetic_problem(7, [("add", 4)])),
        _example(0, 2, _arithmetic_problem(4, [("subtract", 6), ("multiply", 3)])),
    ],
    1: [
        _example(1, 1, _pointer_problem(7, _cycle([0, 4, 9, 1, 7, 3, 8, 2, 6, 5]), 1)),
        _example(1, 2, _pointer_problem(4, _cycle([0, 8, 4, 7, 2, 9, 5, 1, 6, 3]), 2)),
    ],
}
DEMO_IDS = {example["id"] for examples in DEMOS.values() for example in examples}


def format_prompt(example: dict[str, Any]) -> str:
    """Build a prompt from target task/problem only; gold fields are never read."""
    task_id = int(example["task_id"])
    parts = ["Solve the problem. Show the starting digit and each next digit. End with a line: Answer: d"]
    for demo in DEMOS[task_id]:
        path = " -> ".join(map(str, demo["gold_trace"]))
        parts.append(f"Problem: {demo['problem']}\nWork: {path}\nAnswer: {demo['answer']}")
    parts.append(f"Problem: {example['problem']}\nWork:")
    return "\n\n".join(parts)


def extract_answer(text: str) -> str | None:
    """Accept exactly one final single-digit Answer line; reject e.g. Answer: 16."""
    # Count malformed markers too: a second attempt must not rescue an earlier
    # multi-digit or otherwise invalid answer.
    if len(re.findall(r"\bAnswer\s*:", text, re.IGNORECASE)) != 1:
        return None
    lines = text.strip().splitlines()
    matches = [(i, re.fullmatch(r"\s*Answer\s*:\s*([0-9])\s*[.!]?\s*", line, re.IGNORECASE)) for i, line in enumerate(lines)]
    matches = [(i, match) for i, match in matches if match is not None]
    if len(matches) != 1 or matches[0][0] != len(lines) - 1:
        return None
    return matches[0][1].group(1)


def extract_trace(text: str) -> list[int] | None:
    """Diagnostic only. Read a single arrow-separated Work line, not the answer."""
    for line in text.splitlines():
        candidate = re.sub(r"^\s*Work\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if re.fullmatch(r"[0-9](?:\s*->\s*[0-9])+", candidate):
            return [int(token.strip()) for token in candidate.split("->")]
    return None


def _ordinary_split(example: dict[str, Any]) -> str:
    if example["difficulty"] >= 5:
        return "ood"
    bucket = int(_digest(f"{SPLIT_VERSION}|bucket|{example['id']}")[:8], 16) % 100
    return "train" if bucket < 60 else ("dev" if bucket < 80 else "test")


def _one_step_pool() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Exhaust the finite one-step arithmetic domain, then stratify by hash rank.

    A modulo hash bucket can accidentally starve one answer in this tiny domain.
    Ranking by content hash within each answer gives deterministic disjoint strata
    with enough examples for all ten answers. No adaptation seed affects strata.
    """
    pool = {split: {str(answer): [] for answer in range(10)} for split in ("train", "dev", "test")}
    for answer in range(10):
        examples = []
        for operation in OPERATIONS:
            start = _reverse(answer, operation)
            example = _example(0, 1, _arithmetic_problem(start, [operation]))
            if example["id"] not in DEMO_IDS:
                examples.append(example)
        examples.sort(key=lambda row: _digest(f"{SPLIT_VERSION}|rank|{row['id']}"))
        train_end = 3 * len(examples) // 5
        dev_end = train_end + len(examples) // 5
        for split, selected in (("train", examples[:train_end]), ("dev", examples[train_end:dev_end]), ("test", examples[dev_end:])):
            pool[split][str(answer)] = selected
    return pool


def _candidate(rng: random.Random, task_id: int, difficulty: int, answer: int) -> dict[str, Any]:
    if task_id == 0:
        operations = [rng.choice(OPERATIONS) for _ in range(difficulty)]
        start = answer
        for operation in reversed(operations):
            start = _reverse(start, operation)
        return _example(0, difficulty, _arithmetic_problem(start, operations))
    values = list(range(10))
    rng.shuffle(values)
    mapping = _cycle(values)
    inverse = {value: key for key, value in mapping.items()}
    start = answer
    for _ in range(difficulty):
        start = inverse[start]
    return _example(1, difficulty, _pointer_problem(start, mapping, difficulty))


def make_data(seed: int = 17, train_n: int = 200, dev_n: int = 40, test_n: int = 100,
              ood_n: int = 100, n_per_level: int | None = None) -> dict[str, Any]:
    """Create balanced examples per task, per difficulty, for four disjoint splits.

    Counts must be multiples of ten. Defaults are per task per difficulty, not
    overall counts. Train/dev/test use levels 1--4; OOD uses levels 5--7. Only
    one-step arithmetic is capped (80/20/20), reflecting its finite 210-problem
    domain. Training sampling depends on ``seed``; held-out sampling does not.
    Additional ``diagnostics`` documents actual counts and split construction.
    """
    if n_per_level is not None:
        train_n = dev_n = test_n = ood_n = n_per_level
    requested = {"train": train_n, "dev": dev_n, "test": test_n, "ood": ood_n}
    for split, count in requested.items():
        if not isinstance(count, int) or count < 0 or count % 10:
            raise ValueError(f"{split} count must be a nonnegative multiple of 10, got {count}")
    one_step = _one_step_pool()
    data: dict[str, Any] = {split: [] for split in requested}
    seen = set(DEMO_IDS)
    counts = []
    for split, requested_count in requested.items():
        sampling_seed = seed if split == "train" else EVALUATION_SEED
        levels = (5, 6, 7) if split == "ood" else (1, 2, 3, 4)
        for task_id in (0, 1):
            for difficulty in levels:
                count = min(requested_count, ONE_STEP_CAPS[split]) if task_id == 0 and difficulty == 1 else requested_count
                before = len(data[split])
                for answer in range(10):
                    rng = random.Random(_seed(SPLIT_VERSION, sampling_seed, split, task_id, difficulty, answer))
                    target = count // 10
                    if task_id == 0 and difficulty == 1:
                        candidates = list(one_step[split][str(answer)])
                        rng.shuffle(candidates)
                        if target > len(candidates):
                            raise ValueError("Finite one-step stratum cannot meet requested balanced count")
                        selected = candidates[:target]
                        for example in selected:
                            if example["id"] in seen:
                                raise AssertionError("Duplicate or demonstration in split")
                            seen.add(example["id"])
                            data[split].append(example)
                    else:
                        added = attempts = 0
                        while added < target:
                            attempts += 1
                            if attempts > max(10000, 2000 * target):
                                raise ValueError(f"Requested count exceeds practical domain capacity: {split}, task {task_id}, level {difficulty}")
                            example = _candidate(rng, task_id, difficulty, answer)
                            if example["id"] in seen or _ordinary_split(example) != split:
                                continue
                            seen.add(example["id"])
                            data[split].append(example)
                            added += 1
                actual = len(data[split]) - before
                counts.append({"split": split, "task_id": task_id, "difficulty": difficulty,
                               "requested": requested_count, "actual": actual, "per_answer": actual // 10})
        random.Random(_seed(SPLIT_VERSION, sampling_seed, split, "order")).shuffle(data[split])
    data["diagnostics"] = {
        "split_version": SPLIT_VERSION,
        "training_seed": seed,
        "heldout_seed": EVALUATION_SEED,
        "heldout_fixed_across_training_seeds": True,
        "counts": counts,
        "one_step_arithmetic_domain_size": 210,
        "one_step_arithmetic_caps": ONE_STEP_CAPS,
        "split_method": "Fixed 60/20/20 content-hash buckets; finite one-step arithmetic uses answer-stratified content-hash ranks; OOD has separate difficulties.",
        "answer_balance": "Exactly uniform within every task/difficulty/split.",
        "excluded_demonstration_ids": sorted(DEMO_IDS),
        "caveat": "A split holds out exact problems, not operation primitives or constituent map edges. One-step arithmetic has deliberately smaller datasets.",
    }
    return data


def self_test() -> dict[str, Any]:
    data = make_data(17, train_n=100, dev_n=30, test_n=40, ood_n=30)
    all_ids: set[str] = set()
    examples_checked = 0
    for split in ("train", "dev", "test", "ood"):
        strata: dict[tuple[int, int], Counter] = {}
        for example in data[split]:
            assert example["id"] not in all_ids | DEMO_IDS
            all_ids.add(example["id"])
            assert solve(example["task_id"], example["problem"]) == {"answer": example["answer"], "gold_trace": example["gold_trace"]}
            assert len(example["gold_trace"]) == example["difficulty"] + 1
            strata.setdefault((example["task_id"], example["difficulty"]), Counter())[example["answer"]] += 1
            poisoned = dict(example, answer="SECRET_LEAK", gold_trace=["SECRET_LEAK"])
            assert format_prompt(poisoned) == format_prompt(example)
            assert "SECRET_LEAK" not in format_prompt(poisoned)
            examples_checked += 1
        for counts in strata.values():
            assert set(counts) == set(map(str, range(10))) and len(set(counts.values())) == 1
    repeat = make_data(17, train_n=100, dev_n=30, test_n=40, ood_n=30)
    assert data == repeat
    other_seed = make_data(31, train_n=100, dev_n=30, test_n=40, ood_n=30)
    for split in ("dev", "test", "ood"):
        assert data[split] == other_seed[split]
    old_eval = {example["id"] for split in ("dev", "test", "ood") for example in data[split]}
    assert not old_eval & {example["id"] for example in other_seed["train"]}
    assert extract_answer("1 -> 7\nAnswer: 7") == "7"
    assert extract_answer("Answer: 16") is None
    assert extract_answer("Answer: 7\nAnswer: 3") is None
    assert extract_answer("Answer: 16\nAnswer: 2") is None
    assert extract_answer("Answer: unknown\nANSWER: 2") is None
    assert extract_answer("Earlier Answer: 16\nAnswer: 2") is None
    assert extract_answer("Answer: 7\nextra") is None
    assert extract_answer("Work: 1 -> 2\nANSWER: 2.") == "2"
    assert extract_trace("Work: 1 -> 2 -> 4\nAnswer: 4") == [1, 2, 4]
    return {"passed": True, "examples_checked": examples_checked,
            "split_counts": {split: len(data[split]) for split in ("train", "dev", "test", "ood")},
            "tests": ["balanced answers", "solver consistency", "disjoint splits", "excluded demonstrations",
                      "prompt does not read targets", "repeatability", "fixed held-out sets across seeds", "strict answer parser"]}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
