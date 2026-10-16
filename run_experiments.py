#!/usr/bin/env python3
"""DAI 2027 experiment entrypoint: six independent GPU workers and Drive recovery.

See README.md for installation and headless Google Drive authorization.
No distributed-training service, notebook runtime, or GPU interconnect is used.
"""
import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
RECIPES = ("unrouted", "routed", "unrouted_preserve_all", "routed_preserve_all")
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def plan_for(args):
    files = [Path(__file__), ROOT / "requirements.txt", *sorted(SRC.glob("*.py"))]
    recipes = ("routed_preserve_all",) if args.smoke else RECIPES
    seeds = [31] if args.smoke else args.seeds
    jobs = [dict(support=support, recipe=recipe, seed=seed,
                 id=f"{support}__{recipe}__s{seed}")
            for seed in seeds for support in ("native", "lora") for recipe in recipes]
    random.Random(20261016).shuffle(jobs)
    return dict(study="dai2027-native-vs-lora-v1", model=MODEL, revision=REVISION,
                precision="float32", trajectory_microbatch=1, lora_rank=18, lora_alpha=18,
                lr=args.lr, steps=2 if args.smoke else args.steps,
                prompts_per_update=4, group_size=4, response_cap=96,
                save_every=1 if args.smoke else 20, evaluation="all four masks; final is primary",
                smoke=bool(args.smoke),
                advantage_mode="scripted +1 diagnostic" if args.smoke else "binary-correctness RLOO",
                jobs=jobs,
                source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in files})


def read_or_write_plan(output, plan):
    target = output / "experiment_plan.json"
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError("Output directory has different scientific settings/source; use a new run directory")
    atomic_json(target, plan)


def dependencies():
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Use Python 3.10–3.12; the setup instructions use Python 3.11")
    try:
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("Install requirements.txt in your virtual environment first") from error
    if torch.__version__.split("+")[0] != "2.6.0" or transformers.__version__ != "4.51.3":
        raise RuntimeError("This study is pinned to torch 2.6.0 and transformers 4.51.3")
    return torch


def worker(args, plan):
    import torch
    import nr6_run as runner
    import nr6_train as train
    import nr6_io as io
    import nr5_rl as v5
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dai_model import LoRASlots
    from dai_drive import DriveBackup
    from dai_rl import update

    spec = next(j for j in plan["jobs"] if j["id"] == args._worker)
    job_id = spec["id"]
    fit_root = args.output / "fits" / job_id
    os.chdir(SRC)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Worker has no CUDA GPU")
    torch.cuda.set_device(0)
    train.RECIPES.update({"unrouted": ("unrouted", "all_ab"),
                          "routed": ("routed", "all_ab"),
                          "unrouted_preserve_all": ("unrouted_preserve", "all_ab")})
    if args.smoke:
        def smoke_update(module, tok, items, advantages, config, optimizers, task, routing, pitems):
            # Plumbing tests must exercise a nonzero update even when reward is zero.
            result = update(module, tok, items, [1.] * len(items), config,
                            optimizers, task, routing, pitems)
            result["smoke_scripted_advantages"] = True
            return result
        train.extended_update = smoke_update
    else:
        train.extended_update = update
    runner.SOURCE_FILES += ["dai_model.py", "dai_rl.py", "dai_drive.py"]
    original_runtime = io.runtime_info
    io.runtime_info = lambda: dict(original_runtime(), dtype="float32", microbatch=1)

    def load_model():
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left",
                                            trust_remote_code=False, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            use_safetensors=True, attn_implementation="sdpa", trust_remote_code=False,
            local_files_only=True).to("cuda").eval()
        return model, tok

    runner.load_model = load_model
    if spec["support"] == "lora":
        runner.NativeSlots = LoRASlots
    original_predict = v5.predict

    def predict(*positional, **keywords):
        rows = original_predict(*positional, **keywords)
        for row in rows:
            row.update(support=spec["support"], cluster_job=job_id)
        return rows

    v5.predict = predict
    if args.smoke:
        original_make_data = runner.make_data

        def small_data(*positional, **keywords):
            data = original_make_data(*positional, **keywords)
            for split in ("train", "dev", "test", "depth_probe"):
                counts, chosen = {}, []
                for row in data[split]:
                    key = row["task_id"], row["difficulty"]
                    counts[key] = counts.get(key, 0) + 1
                    if counts[key] <= 2:
                        chosen.append(row)
                data[split] = chosen
            data["provenance"]["cluster_smoke_subset"] = "first two rows per task/depth; plumbing only"
            return data

        runner.make_data = small_data

    @dataclass
    class ClusterConfig(train.Config):
        cluster_plan_sha256: str = ""
        base_dtype: str = "float32"
        trajectory_microbatch: int = 1

    cfg = ClusterConfig(max_steps=plan["steps"], lr=plan["lr"], max_new_tokens=96,
                        max_prompt_tokens=768, eval_every=1 if args.smoke else 80,
                        save_every=plan["save_every"], generation_batch=2, eval_batch=2,
                        head_chunk=8, kl_head_chunk=8, audit_steps=[1, 2] if args.smoke else [],
                        cluster_plan_sha256=io.digest(plan)).validate()
    ws, data, probes, jobs = runner.prepare(fit_root / "local", fit_root / "durable", cfg,
                                           [spec["seed"]], [spec["recipe"]], families=2,
                                           smoke=args.smoke)
    rel = jobs[0]["job"]
    drive = DriveBackup(args.drive_remote)
    drive.restore(job_id, ws.durable, f"{rel}/continuation.pt", ws.run_hash)
    registry = ws.durable / "artifacts.json"
    if registry.exists():
        ws.artifacts.update(json.loads(registry.read_text()))
    original_save = ws.save_training
    original_publish = ws.publish

    def publish(path, evidence=True):
        original_publish(path, evidence=evidence)
        if "/evaluation/" in str(path) or str(path).endswith("predictions_final.json"):
            drive.upload_file(ws.durable / path, f"{job_id}/evidence/{path}")

    def save(job, config):
        original_save(job, config)
        resume = json.loads(ws.path(f"{job}/resume.json").read_text())
        drive.mirror_evidence(job_id, ws.durable)
        drive.commit(job_id, ws.durable / job / "continuation.pt", resume["step"], ws.run_hash)

    ws.save_training = save
    ws.publish = publish
    result = runner.run_jobs(ws, data, probes, cfg, jobs, diagnose_new=False, count_hint=False)
    evidence_zip = ws.export()
    drive.mirror_evidence(job_id, ws.durable)
    completion = dict(spec, plan_sha256=io.digest(plan), result=result,
                      complete=json.loads((ws.durable / rel / "complete.json").read_text()),
                      metrics=json.loads((ws.durable / rel / "metrics.json").read_text()),
                      audits=json.loads((ws.durable / rel / "checkpoint_audits.json").read_text()),
                      evidence_zip=str(evidence_zip.relative_to(args.output)))
    marker = fit_root / "completed.json"
    pending_marker = fit_root / "completed.pending.json"
    atomic_json(pending_marker, completion)
    drive.upload_file(pending_marker, f"{job_id}/completed.json")
    os.replace(pending_marker, marker)
    print(f"COMPLETE AND BACKED UP: {job_id}", flush=True)


def collect(output, plan):
    rows = []
    for spec in plan["jobs"]:
        p = output / "fits" / spec["id"] / "completed.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    summary = dict(planned=len(plan["jobs"]), completed=len(rows), fits=rows,
                   primary_endpoint="final", smoke=plan["smoke"])
    atomic_json(output / "summary.json", summary)
    import csv
    with (output / "final_results.csv").open("w", newline="") as stream:
        fields = ["support", "recipe", "seed", "task", "split", "n", "base", "owner", "all_on",
                  "foreign", "composition_lost", "composition_rescued", "foreign_changed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fit in rows:
            grouped = {}
            for row in fit["metrics"]["final"]:
                key = row["task_id"], row["split"]
                target = grouped.setdefault(key, {k: 0 for k in fields[5:]})
                for field, source in [("n", "n"), ("base", "none_correct"), ("owner", "owner_correct"),
                                      ("all_on", "AB_correct"), ("foreign", "foreign_correct"),
                                      ("composition_lost", "composition_lost"),
                                      ("composition_rescued", "composition_rescued"),
                                      ("foreign_changed", "foreign_changed")]:
                    target[field] += row[source]
            for (task, split), scores in grouped.items():
                writer.writerow(dict(support=fit["support"], recipe=fit["recipe"], seed=fit["seed"],
                                     task="arithmetic" if task == 0 else "pointer", split=split, **scores))
    return len(rows)


def launch(args, plan):
    torch = dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected; check nvidia-smi and the PyTorch CUDA installation")
    if not shutil.which("rclone"):
        raise RuntimeError("Install rclone and configure the gdrive remote first")
    if args.gpus:
        devices = [value.strip() for value in args.gpus.split(",")]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    if not devices or len(set(devices)) != len(devices) or any(not d for d in devices):
        raise ValueError("GPU identifiers must be nonempty and unique")
    from dai_drive import DriveBackup
    backup = DriveBackup(args.drive_remote)
    backup.ensure_plan(plan)
    print("Running offline checks before downloading/loading the model...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tests" / "test_study.py")], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = []
    for spec in plan["jobs"]:
        marker = args.output / "fits" / spec["id"] / "completed.json"
        names = backup._names(spec["id"])
        if "completed.json" in names:
            remote = json.loads(backup._call("cat", backup._target(f"{spec['id']}/completed.json")))
            from nr6_io import digest
            if remote.get("plan_sha256") != digest(plan):
                raise ValueError("Completed Drive fit has a different experiment plan")
            atomic_json(marker, remote)
        if marker.exists():
            from nr6_io import digest
            if json.loads(marker.read_text()).get("plan_sha256") != digest(plan):
                raise ValueError("Local completed fit has a different experiment plan")
            print("Already complete:", spec["id"], flush=True)
        else:
            pending.append(spec)
    processes, failed = {}, []
    available = list(devices)
    last_status = 0.
    try:
        while pending or processes:
            while pending and available:
                spec, device = pending.pop(0), available.pop(0)
                logfile = logs / f"{spec['id']}.log"
                stream = logfile.open("a", buffering=1)
                command = [sys.executable, str(Path(__file__).resolve()), "--_worker", spec["id"],
                           "--output", str(args.output), "--drive-remote", args.drive_remote,
                           "--steps", str(args.steps), "--lr", str(args.lr),
                           "--seeds", ",".join(map(str, args.seeds))]
                if args.smoke:
                    command.append("--smoke")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, TOKENIZERS_PARALLELISM="false",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                processes[process.pid] = (process, device, spec, stream)
                print(f"START GPU {device}: {spec['id']} -> {logfile}", flush=True)
            for pid, (process, device, spec, stream) in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del processes[pid]
                available.append(device)
                if code:
                    failed.append(dict(job=spec["id"], exit_code=code))
                    print(f"FAILED {spec['id']}: see {logs / (spec['id'] + '.log')}", flush=True)
                else:
                    print("FINISHED:", spec["id"], flush=True)
                collect(args.output, plan)
            if time.monotonic() - last_status >= 30:
                complete = collect(args.output, plan)
                print(f"Progress: {complete}/{len(plan['jobs'])} complete; "
                      f"{len(processes)} running; {len(pending)} queued; {len(failed)} failed", flush=True)
                last_status = time.monotonic()
            time.sleep(2)
    finally:
        for process, _, _, _ in processes.values():
            process.send_signal(signal.SIGINT)
        for process, _, _, stream in processes.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        complete = collect(args.output, plan)
        backup.upload_file(args.output / "summary.json", "summary.json")
        backup.upload_file(args.output / "final_results.csv", "final_results.csv")
    if failed or complete != len(plan["jobs"]):
        atomic_json(args.output / "failed_jobs.json", failed)
        raise RuntimeError("Some fits remain incomplete. Fix the logged error, then repeat the same command to resume.")
    print(f"DONE: {complete} fits. Results: {args.output / 'final_results.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/paper01"))
    parser.add_argument("--drive-remote", help="Configured rclone remote plus a dedicated run folder")
    parser.add_argument("--gpus", help="Comma-separated visible GPU IDs; default: all available GPUs")
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[31, 47, 71])
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Two small real-GPU fits to check setup and backups")
    parser.add_argument("--plan", action="store_true", help="Print the experiment plan without running anything")
    parser.add_argument("--self-test", action="store_true", help="Run offline numerical and recovery checks")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 2 or args.lr <= 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least two updates, positive learning rate, and distinct seeds")
    if args.self_test:
        dependencies()
        subprocess.run([sys.executable, str(ROOT / "tests" / "test_study.py")], check=True)
        return
    plan = plan_for(args)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not args.drive_remote:
        parser.error("--drive-remote is required: checkpoint backup is part of this runner")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args._worker:
        dependencies()
        if json.loads((args.output / "experiment_plan.json").read_text()) != plan:
            raise ValueError("Worker plan differs from its launcher")
        worker(args, plan)
    else:
        with (args.output / ".run.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("A launcher already owns this output directory") from error
            read_or_write_plan(args.output, plan)
            launch(args, plan)


if __name__ == "__main__":
    main()
