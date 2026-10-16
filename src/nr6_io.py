"""Bounded, explicit-path Drive I/O; all training/evaluation runs on local disk."""
from pathlib import Path
import hashlib, json, os, platform, shutil, threading, time, zipfile
import torch, transformers
from nr5_rl import atomic_json, atomic_torch, _load

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def runtime_info():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
        transformers=transformers.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        dtype='bfloat16' if torch.cuda.is_available() else 'float32')

def atomic_copy(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve(): return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.partial')
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(15):
            print(f'  Still copying {source.name} ({source.stat().st_size/2**20:.1f} MiB)...', flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        shutil.copyfile(source, temporary)
        assert temporary.stat().st_size == source.stat().st_size
        os.replace(temporary, target)
    finally:
        finished.set(); thread.join(timeout=1)

class Workspace:
    def __init__(self, local, durable, identity):
        self.local, self.durable = Path(local), Path(durable)
        self.local.mkdir(parents=True, exist_ok=True); self.durable.mkdir(parents=True, exist_ok=True)
        self.identity = identity; self.run_hash = digest(identity)
        print('Validating this run manifest (no recursive folder scan)...', flush=True)
        manifest = self.durable/'manifest.json'
        if manifest.exists():
            assert json.loads(manifest.read_text()) == identity, 'Scientific settings/source changed: choose a new RUN_TAG.'
        else: atomic_json(manifest, identity)
        atomic_json(self.local/'manifest.json', identity)
        registry = self.durable/'artifacts.json'
        self.artifacts = set(json.loads(registry.read_text())) if registry.exists() else {'manifest.json'}
        self.runtime = runtime_info(); self.runtime_key = digest(self.runtime)[:16]
        sessions_path = self.durable/'runtime_sessions.json'
        sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
        changed = bool(sessions and sessions[-1]['runtime'] != self.runtime)
        sessions.append(dict(time=time.time(), runtime=self.runtime, changed=changed))
        self.write_json('runtime_sessions.json', sessions)
        if changed: print('Hardware/software runtime changed: training resume is allowed; evaluation gets a separate cache.', flush=True)

    def path(self, rel):
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts
        return self.local/p

    def publish(self, rel, evidence=True):
        rel = str(Path(rel)); p = self.path(rel)
        atomic_copy(p, self.durable/rel)
        if evidence and rel not in self.artifacts:
            self.artifacts.add(rel)
            atomic_json(self.durable/'artifacts.json', sorted(self.artifacts))
            atomic_json(self.local/'artifacts.json', sorted(self.artifacts))

    def write_json(self, rel, value):
        atomic_json(self.path(rel), value); self.publish(rel)

    def pull(self, rel):
        p = self.path(rel); src = self.durable/rel
        if src.exists():
            atomic_copy(src, p); return p
        return p if p.exists() else None

    def save_training(self, job, cfg):
        """One atomic file holds the mutually consistent latest AND best states.

        An interruption during copy leaves the previous complete bundle intact.
        No archive walk, model reload, or GPU check is done here.
        """
        folder = self.path(job)
        latest = _load(folder/'latest.pt')
        best = _load(folder/'best.pt') if (folder/'best.pt').exists() else None
        audits = {str(s): json.loads((folder/'update_audits'/f'{s:05d}.json').read_text())
                  for s in cfg.audit_steps if s <= latest['step'] and (folder/'update_audits'/f'{s:05d}.json').exists()}
        dev = {str(s): json.loads((folder/'dev_predictions'/f'step_{s:05d}.json').read_text())
               for s in range(0, latest['step']+1) if (s % cfg.eval_every == 0 or s == cfg.max_steps)
               and (folder/'dev_predictions'/f'step_{s:05d}.json').exists()}
        bundle = dict(run_hash=self.run_hash, latest=latest, best=best, audits=audits, dev_predictions=dev)
        atomic_torch(folder/'continuation.pt', bundle)
        print(f'Saving {job}: step {latest["step"]}, pending_dev={latest["pending_dev"]}', flush=True)
        self.publish(str(Path(job)/'continuation.pt'), evidence=False)
        for name in ('config.json', 'history.jsonl', 'dev_history.json'):
            if (folder/name).exists(): self.publish(str(Path(job)/name))
        for s in audits: self.publish(str(Path(job)/'update_audits'/f'{int(s):05d}.json'))
        for s in dev: self.publish(str(Path(job)/'dev_predictions'/f'step_{int(s):05d}.json'))
        self.write_json(str(Path(job)/'resume.json'), dict(step=latest['step'], best_step=latest['best_step'],
            pending_dev=latest['pending_dev'], runtime=self.runtime, run_hash=self.run_hash))

    def restore_training(self, job, cfg):
        folder = self.path(job); folder.mkdir(parents=True, exist_ok=True)
        source = self.durable/job/'continuation.pt'
        if not source.exists(): return False
        print(f'Restoring one checkpoint bundle for {job}...', flush=True)
        self.pull(str(Path(job)/'continuation.pt'))
        bundle = _load(folder/'continuation.pt')
        assert bundle['run_hash'] == self.run_hash
        latest, best = bundle['latest'], bundle['best']
        if best is not None:
            assert best['step'] == latest['best_step'] and best['signature'] == latest['signature']
        atomic_torch(folder/'latest.pt', latest)
        if best is not None: atomic_torch(folder/'best.pt', best)
        elif (folder/'best.pt').exists(): (folder/'best.pt').unlink()
        atomic_json(folder/'config.json', latest['signature'])
        for s, value in bundle['audits'].items(): atomic_json(folder/'update_audits'/f'{int(s):05d}.json', value)
        for s, value in bundle['dev_predictions'].items(): atomic_json(folder/'dev_predictions'/f'step_{int(s):05d}.json', value)
        for s in cfg.audit_steps:
            p = folder/'update_audits'/f'{s:05d}.json'
            if s > latest['step'] and p.exists(): p.unlink()
        # A status from another local attempt must never override the committed bundle.
        status = folder/'status.json'
        if status.exists(): status.unlink()
        print(f'Restored step {latest["step"]}; best step {latest["best_step"]}.', flush=True)
        return True

    def export(self):
        """Only registered evidence paths; never recurse through Google Drive."""
        target = self.local/'diagnostics_v6.zip'; temporary = target.with_suffix('.zip.partial')
        print(f'Exporting {len(self.artifacts)} registered evidence files; tensor checkpoints stay in the run folder.', flush=True)
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as out:
            for rel in sorted(self.artifacts):
                p = self.path(rel)
                if not p.exists(): self.pull(rel)
                if p.exists(): out.write(p, rel)
        os.replace(temporary, target)
        atomic_copy(target, self.durable/target.name)
        return target
