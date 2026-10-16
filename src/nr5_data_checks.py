"""CPU-only data and prompt contract checks, with no model downloads."""
import json
import nr5_data
import nr5_tasks


def run_checks():
    result = nr5_data.self_test()
    rows = nr5_data.make_data("mixed", "smoke")["test"]
    for row in rows:
        stripped = {k: v for k, v in row.items() if k not in ("answer", "gold_trace")}
        assert nr5_tasks.format_prompt(row) == nr5_tasks.format_prompt(stripped)
        for variant in nr5_tasks.VARIANTS:
            assert nr5_tasks.variant_prompt(row, variant) == nr5_tasks.variant_prompt(stripped, variant)
    family = next(row for row in rows if row["task_id"] == 1)
    shifted = nr5_tasks.with_pointer_depth(family, 5)
    assert nr5_tasks.pointer_family_key(shifted) == nr5_tasks.pointer_family_key(family)
    assert len(shifted["gold_trace"]) == 6
    diagnostics = nr5_tasks.build_pointer_counterfactuals(nfamilies=20)
    assert diagnostics["provenance"]["n_maps"] == 20
    assert diagnostics["provenance"]["start_counts"] == {str(d): 2 for d in range(10)}
    assert nr5_tasks.build_pointer_counterfactuals(nfamilies=7)["provenance"]["n_maps"] == 7
    result["targetblind_variants"] = True
    return result


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
