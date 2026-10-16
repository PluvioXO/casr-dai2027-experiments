"""Committed, content-addressed checkpoints on Google Drive through rclone.

A checkpoint becomes a restore candidate only after its blob upload and remote
MD5 verification succeed. Its SHA-256 is checked again after download. Keep the
two newest committed checkpoints; never delete anything outside this job's
checkpoint namespace. Evaluation caches and logs are copied without deletion.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def file_hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


class DriveBackup:
    def __init__(self, remote):
        if ":" not in remote or any(c in remote for c in "\r\n"):
            raise ValueError("Use an rclone destination such as gdrive:CASR_DAI2027/run01")
        self.remote = remote.rstrip("/")

    def _call(self, *args):
        result = subprocess.run(["rclone", *map(str, args), "--retries", "3",
                                 "--low-level-retries", "10", "--contimeout", "30s",
                                 "--timeout", "5m", "--stats", "0", "--log-level", "ERROR"],
                                capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError(f"rclone {args[0]} failed: {result.stderr.strip()}")
        return result.stdout

    def _target(self, rel):
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Backup paths must stay inside the selected run folder")
        return self.remote + "/" + path.as_posix()

    def _names(self, rel):
        self._call("mkdir", self._target(rel))
        rows = json.loads(self._call("lsjson", self._target(rel), "--files-only", "--max-depth", "1"))
        return [r["Name"] for r in rows]

    def upload_file(self, local, rel):
        self._call("copyto", local, self._target(rel), "--checksum")

    def ensure_plan(self, plan):
        self._call("mkdir", self.remote)
        rows = json.loads(self._call("lsjson", self.remote, "--files-only", "--max-depth", "1"))
        if any(r["Name"] == "experiment_plan.json" for r in rows):
            old = json.loads(self._call("cat", self._target("experiment_plan.json")))
            if old != plan:
                raise ValueError("Drive folder belongs to different settings/source. Choose a new run folder.")
        else:
            with tempfile.TemporaryDirectory(prefix="casr_drive_plan_") as folder:
                p = Path(folder) / "experiment_plan.json"
                p.write_text(json.dumps(plan, indent=2, sort_keys=True))
                self.upload_file(p, "experiment_plan.json")
                if json.loads(self._call("cat", self._target("experiment_plan.json"))) != plan:
                    raise RuntimeError("Drive plan readback failed")

    def records(self, job, run_hash):
        names = self._names(f"{job}/commits")
        records = []
        for name in names:
            if not re.fullmatch(r"[0-9]{5}-[0-9]{19}-[0-9a-f]{16}\.json", name):
                continue
            record = json.loads(self._call("cat", self._target(f"{job}/commits/{name}")))
            sha = record.get("sha256", "")
            if record.get("run_hash") != run_hash:
                raise ValueError("Checkpoint manifest identity differs from this run")
            if (not re.fullmatch(r"[0-9a-f]{64}", sha)
                    or record.get("blob") != f"checkpoints/{int(record['step']):05d}/{sha}.pt"):
                raise ValueError("Malformed checkpoint commit record")
            records.append((name, record))
        return sorted(records, key=lambda row: (row[1]["step"], row[1]["created_ns"]), reverse=True)

    def commit(self, job, path, step, run_hash):
        path = Path(path)
        sha, md5 = file_hashes(path)
        existing = self.records(job, run_hash)
        if existing and existing[0][1]["sha256"] == sha:
            return
        blob = f"checkpoints/{step:05d}/{sha}.pt"
        print(f"Drive backup: {job}, step {step}, {path.stat().st_size / 2**20:.1f} MiB", flush=True)
        self._call("copyto", path, self._target(f"{job}/{blob}"), "--checksum", "--immutable")
        remote_md5 = self._call("md5sum", self._target(f"{job}/{blob}")).split()[0]
        if remote_md5.lower() != md5:
            raise RuntimeError("Uploaded checkpoint checksum mismatch; previous commits retained")
        created = time.time_ns()
        record = dict(version=1, step=int(step), run_hash=run_hash, sha256=sha,
                      md5=md5, bytes=path.stat().st_size, blob=blob, created_ns=created)
        name = f"{step:05d}-{created:019d}-{sha[:16]}.json"
        with tempfile.TemporaryDirectory(prefix="casr_drive_commit_") as folder:
            p = Path(folder) / name
            p.write_text(json.dumps(record, sort_keys=True))
            self.upload_file(p, f"{job}/commits/{name}")
            if json.loads(self._call("cat", self._target(f"{job}/commits/{name}"))) != record:
                raise RuntimeError("Checkpoint commit readback failed; previous commits retained")
        print(f"Drive backup verified: {job}, step {step}", flush=True)
        candidates = sorted([(name, record)] + existing,
                            key=lambda row: (row[1]["step"], row[1]["created_ns"]), reverse=True)
        retained = {r["blob"] for _, r in candidates[:2]}
        for old_name, old in candidates[2:]:
            try:
                self._call("deletefile", self._target(f"{job}/commits/{old_name}"))
                if old["blob"] not in retained:
                    self._call("deletefile", self._target(f"{job}/{old['blob']}"))
            except RuntimeError as error:
                print(f"Checkpoint is safe; old-backup cleanup failed: {error}", flush=True)

    def mirror_evidence(self, job, folder):
        self._call("copy", folder, self._target(f"{job}/evidence"),
                   "--exclude", "*.pt", "--exclude", "*.partial", "--exclude", "*.tmp",
                   "--exclude", "__pycache__/**", "--checksum")

    def restore(self, job, folder, checkpoint_rel, run_hash):
        import torch
        folder = Path(folder)
        records = self.records(job, run_hash)
        if not records:
            return False
        target = folder / checkpoint_rel
        if target.exists():
            try:
                local = torch.load(target, map_location="cpu", weights_only=True)
                if local["run_hash"] == run_hash and local["latest"]["step"] >= records[0][1]["step"]:
                    print("Using the local committed checkpoint; Drive remains the recovery copy.", flush=True)
                    return True
            except (OSError, RuntimeError, EOFError, KeyError, ValueError):
                pass
        self._call("copy", self._target(f"{job}/evidence"), folder,
                   "--exclude", "*.pt", "--exclude", "*.tmp", "--exclude", "*.partial", "--checksum")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".restore.partial")
        for _, record in records:
            self._call("copyto", self._target(f"{job}/{record['blob']}"), temporary, "--checksum")
            sha, _ = file_hashes(temporary)
            if sha != record["sha256"] or temporary.stat().st_size != record["bytes"]:
                print(f"Ignoring damaged Drive checkpoint at step {record['step']}", flush=True)
                continue
            bundle = torch.load(temporary, map_location="cpu", weights_only=True)
            if bundle["run_hash"] != run_hash or bundle["latest"]["step"] != record["step"]:
                raise ValueError("Downloaded checkpoint identity/step differs from its commit")
            os.replace(temporary, target)
            print(f"Recovered from Drive: {job}, step {record['step']}", flush=True)
            return True
        raise RuntimeError("No valid Drive checkpoint could be restored; local files were not overwritten")
