"""Offline checks for accumulation, LoRA trainability, and committed recovery.

Tiny random models do not establish pretrained-model learning or GPU memory use.
The backup integration uses rclone's local backend, not a Google account.
"""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import nr5_rl as v5
from nr5_rl_checks import fixture, scripted_items
from nr4_model import NativeSlots
from nr6_train import Config, extended_update, train_recipe
from nr6_checks import equal, run_checks
from dai_model import LoRASlots
from dai_rl import update
from dai_drive import DriveBackup


def numerical_checks():
    torch.set_num_threads(1)
    base, tok, examples = fixture()
    items = scripted_items(tok, [examples[0]] * 4)
    cfg = Config(max_steps=2, prompts_per_update=1, group_size=4, generation_batch=4,
                 max_prompt_tokens=2048, max_new_tokens=4, eval_every=1, save_every=1,
                 eval_batch=2, lr=1e-3, min_lr=1e-5, max_mean_kl=100,
                 max_sequence_kl=100, checkpointing=True, kl_positions=2,
                 head_chunk=8, kl_head_chunk=7, clip_grad=1e6,
                 preservation_mode="all_ab", audit_steps=[])
    for cls in (NativeSlots, LoRASlots):
        a = cls(copy.deepcopy(base), "mlp", seed=17, n_layers=2, mlp_neurons=3)
        b = cls(copy.deepcopy(base), "mlp", seed=17, n_layers=2, mlp_neurons=3)
        if cls is LoRASlots:
            a.zero_()
            assert any(torch.count_nonzero(p) for name, p, _ in a._named_deltas if name.endswith(".A"))
            batch = v5._training_batch(items, tok.pad_token_id, a.device, cfg)
            with a.gate((True, True)), a.route(0):
                logp, _ = v5._scores(a, batch, cfg)
                (-logp.mean()).backward()
            assert any(p.grad is not None and torch.count_nonzero(p.grad)
                       for name, p, slot in a._named_deltas if slot == 0 and name.endswith(".B"))
            assert all(p.grad is None for p in a.parameters(1))
            for p in a.parameters():
                p.grad = None
        torch.manual_seed(123)
        with torch.no_grad():
            for p in a.parameters():
                p.add_(torch.randn_like(p) * .005)
        b.load_state_dict(a.state_dict())
        opts_a = [torch.optim.AdamW(a.parameters(k), lr=cfg.lr, weight_decay=0) for k in (0, 1, 0, 1)]
        opts_b = [torch.optim.AdamW(b.parameters(k), lr=cfg.lr, weight_decay=0) for k in (0, 1, 0, 1)]
        dense = extended_update(a, tok, items, [1., -1., 1., -1.], cfg, opts_a, 0,
                                "unrouted_preserve", items)
        accumulated = update(b, tok, items, [1., -1., 1., -1.], cfg, opts_b, 0,
                             "unrouted_preserve", items)
        assert dense["accepted"] and accumulated["accepted"]
        for key in ("policy_loss", "reference_kl", "preservation_kl", "neutral_kl"):
            assert abs(dense[key] - accumulated[key]) < 2e-5, (cls.__name__, key)
        error = max(float((a.state_dict()[name] - p).abs().max()) for name, p in b.state_dict().items())
        assert error < 2e-5, (cls.__name__, "dense/accumulated update", error)
        before = b.state_dict()
        optimizer_before = copy.deepcopy([o.state_dict() for o in opts_b])
        rejected = update(b, tok, items, [1., -1., 1., -1.],
                          replace(cfg, max_mean_kl=1e-20, max_sequence_kl=1e-20, backtracks=0),
                          opts_b, 0, "unrouted_preserve", items)
        assert not rejected["accepted"]
        assert equal(before, b.state_dict()) and equal(optimizer_before, [o.state_dict() for o in opts_b])
        assert b.assert_frozen_base()
        with torch.no_grad(), b.gate((False, False)):
            witness = v5._chosen_log_probs(b, v5._training_batch(items, tok.pad_token_id, b.device, cfg))
        with torch.no_grad(), a.gate((False, False)):
            reference = v5._chosen_log_probs(a, v5._training_batch(items, tok.pad_token_id, a.device, cfg))
        assert torch.equal(witness, reference)
        a.close()
        b.close()
        print(f"PASS {cls.__name__}: accumulated objective/update, routing, rollback, frozen base", flush=True)

    def generate(module, tok, group, cfg, mask="AB", sample=False):
        torch.rand(3)
        return scripted_items(tok, group)

    with tempfile.TemporaryDirectory(prefix="dai_resume_") as folder:
        folder = Path(folder)
        data = {"train": examples, "dev": examples}
        full = LoRASlots(copy.deepcopy(base), seed=31, n_layers=2, mlp_neurons=3)
        first = LoRASlots(copy.deepcopy(base), seed=31, n_layers=2, mlp_neurons=3)
        resumed = LoRASlots(copy.deepcopy(base), seed=31, n_layers=2, mlp_neurons=3)
        with patch("nr6_train.extended_update", side_effect=update), patch.object(v5, "_generate", side_effect=generate):
            train_recipe(full, tok, data, cfg, folder / "full", "routed_preserve_all", 31)

            def disconnect():
                if v5._load(folder / "interrupted/latest.pt")["step"] == 1:
                    raise ConnectionError("Simulated session loss")

            try:
                train_recipe(first, tok, data, cfg, folder / "interrupted", "routed_preserve_all", 31, disconnect)
            except ConnectionError:
                pass
            else:
                raise AssertionError("Expected the simulated disconnect")
            train_recipe(resumed, tok, data, cfg, folder / "interrupted", "routed_preserve_all", 31)
        expected = v5._load(folder / "full/latest.pt")
        actual = v5._load(folder / "interrupted/latest.pt")
        for key in ("deltas", "optimizers", "rng"):
            assert equal(expected[key], actual[key]), key
        for module in (full, first, resumed):
            module.close()
    print("PASS interrupted/resumed LoRA equals uninterrupted LoRA, including Adam and RNG", flush=True)


def backup_checks():
    if not shutil.which("rclone"):
        raise RuntimeError("Install rclone before running the backup integration test")
    with tempfile.TemporaryDirectory(prefix="dai_backup_") as folder:
        root = Path(folder)
        drive = DriveBackup(":local:" + str(root / "remote"))
        plan = {"study": "offline-test", "source": "test"}
        drive.ensure_plan(plan)
        drive.ensure_plan(plan)
        try:
            drive.ensure_plan(dict(plan, source="different"))
        except ValueError:
            pass
        else:
            raise AssertionError("A changed plan must not reuse a backup folder")
        evidence = root / "evidence"
        evidence.mkdir()
        (evidence / "manifest.json").write_text(json.dumps(plan))
        checkpoint = root / "continuation.pt"
        drive.mirror_evidence("job", evidence)
        for step in range(3):
            torch.save(dict(run_hash="same", latest={"step": step}, tensor=torch.arange(4)), checkpoint)
            drive.commit("job", checkpoint, step, "same")
        assert len(drive.records("job", "same")) == 2
        restored = root / "fresh"
        assert drive.restore("job", restored, "fit/continuation.pt", "same")
        assert v5._load(restored / "fit/continuation.pt")["latest"]["step"] == 2
        original_call = drive._call

        def fail_commit(*args):
            if args[0] == "copyto" and "/commits/" in str(args[2]):
                raise RuntimeError("Simulated interruption after blob upload")
            return original_call(*args)

        torch.save(dict(run_hash="same", latest={"step": 3}), checkpoint)
        with patch.object(drive, "_call", side_effect=fail_commit):
            try:
                drive.commit("job", checkpoint, 3, "same")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected upload interruption")
        assert drive.records("job", "same")[0][1]["step"] == 2
        newest = drive.records("job", "same")[0][1]
        (root / "remote/job" / newest["blob"]).write_bytes(b"damaged")
        fallback = root / "fallback"
        drive.restore("job", fallback, "fit/continuation.pt", "same")
        assert v5._load(fallback / "fit/continuation.pt")["latest"]["step"] == 1
    print("PASS rclone integration: commit ordering, two-copy retention, fresh restore, corruption fallback", flush=True)


def pipeline_check():
    import nr6_run as runner
    import nr6_train as train
    base, tok, _ = fixture()
    cfg = Config(max_steps=2, prompts_per_update=1, group_size=4, generation_batch=4,
                 max_prompt_tokens=2048, max_new_tokens=4, eval_every=1, save_every=1,
                 eval_batch=16, max_mean_kl=100, max_sequence_kl=100,
                 kl_positions=2, head_chunk=8, kl_head_chunk=7, audit_steps=[])

    def generate(module, tokenizer, examples, config, mask="AB", sample=False):
        return scripted_items(tokenizer, examples)

    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="dai_pipeline_") as folder:
        try:
            os.chdir(ROOT / "src")
            with patch.dict(train.RECIPES, {"unrouted_preserve_all": ("unrouted_preserve", "all_ab")}), \
                 patch("nr6_train.extended_update", side_effect=update), \
                 patch.object(runner, "load_model", return_value=(copy.deepcopy(base), tok)), \
                 patch.object(runner, "NativeSlots", LoRASlots), \
                 patch.object(v5, "_generate", side_effect=generate), \
                 patch.object(torch.cuda, "reset_peak_memory_stats"), \
                 patch.object(torch.cuda, "max_memory_allocated", return_value=0):
                ws, data, probes, jobs = runner.prepare(Path(folder)/"local", Path(folder)/"durable",
                                                        cfg, [31], ["unrouted_preserve_all"],
                                                        families=2, smoke=True)
                result = runner.run_jobs(ws, data, probes, cfg, jobs, diagnose_new=False, count_hint=False)
                assert result["complete"] == 1
                assert ws.export().exists()
                audit = json.loads((ws.durable/jobs[0]["job"]/"checkpoint_audits.json").read_text())
                assert all(a["all_off_max_logit_diff"] == 0 for a in audit.values())
        finally:
            os.chdir(previous)
    print("PASS LoRA + new unrouted/all-position arm through training, both endpoints, audits, and export", flush=True)


if __name__ == "__main__":
    report = run_checks()
    assert report["all_passed"]
    numerical_checks()
    pipeline_check()
    backup_checks()
    plan = json.loads(subprocess.check_output([sys.executable, str(ROOT / "run_experiments.py"), "--plan"], text=True))
    assert len(plan["jobs"]) == 24 and len({j["id"] for j in plan["jobs"]}) == 24
    assert 8 * 3 * 117 * 896 == 2515968
    assert 8 * 3 * 18 * (896 + 4864) == 2488320
    print("ALL OFFLINE CHECKS PASSED. Real RTX 2080 and Google Drive checks still require --smoke.", flush=True)
