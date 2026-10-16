# Controllable RL post-training: DAI 2027 experiments

The CLI entrypoint is **`run_experiments.py`**. It runs independent jobs on one
machine with six RTX 2080 GPUs and saves resumable progress to Google Drive.
The Python source in `src/` is included by `git clone`; no notebook is required.

## What the full run measures

The paper's two missing comparisons are a LoRA baseline and an unrouted control
with the same all-position preservation as the strongest native-slice recipe.
This runner performs the full factorial comparison:

| Parameterization | Reward gradients | Preservation |
|---|---|---|
| Native MLP slices | Unrouted | Off |
| Native MLP slices | Routed | Off |
| Native MLP slices | Unrouted | All response positions |
| Native MLP slices | Routed | All response positions |
| LoRA | Unrouted | Off |
| LoRA | Routed | Off |
| LoRA | Unrouted | All response positions |
| LoRA | Routed | All response positions |

Three seeds (31, 47, 71) give **24 fits**, with up to six running concurrently.
Each fit uses the same pinned Qwen2.5-0.5B base, eight selected transformer
layers, digit arithmetic and pointer-following tasks, 320 attempted updates,
four prompts per update, four responses per prompt, and a 96-token response cap.
The final checkpoint is the primary endpoint. All four enabled-module masks
are evaluated, including depths 4 and 5. Development-selected results are
retained as a secondary endpoint.

Native slots have **2,515,968 trainable scalars each**. Rank-18 LoRA on the same
gate/up/down projections has **2,488,320 scalars per slot**, 1.1% fewer. Both
keep the pretrained parameters frozen and use separate reward/preservation
Adam histories. LoRA uses alpha=18 and no dropout. The common learning rate is
`1e-4`; this is a controlled first comparison, not a tuned LoRA performance ceiling.

RTX 2080 cards do not support BF16. This version uses an FP32 base and FP32
trainable parameters, accumulating one trajectory at a time. It retains the
16-response update and RLOO groups. Numerical checks compare the accumulated
update with the original dense V6 update. Floating-point accumulation order
differs, so these are fresh matched runs, not bitwise reproductions of BF16 V6.
The default is designed for 8 GiB cards; actual peak memory must be checked by
the GPU smoke run on your machine.

## Install on the GPU machine

Use Linux and Python **3.10–3.12**. The commands below assume the repository has
already been cloned and the current directory contains this README.

```bash
sudo apt-get update
sudo apt-get install -y git tmux rclone python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

The CUDA 11.8 wheel supports the older GPU generation. A working NVIDIA driver
is required; PyTorch supplies its CUDA runtime. The wheel choice follows the
[official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v260).
The runner downloads the pinned model once into the ordinary Hugging Face
cache and reuses that cache across workers. The model is public.

## Connect Google Drive once

```bash
rclone config
```

Create a new remote named **`gdrive`**, choose storage type **`drive`**, and
complete Google's authorization. On a headless server, answer **no** when
asked to use a browser on that machine. Run the authorization command rclone
prints on a computer with a browser, then paste the resulting authorization
response into the **server's rclone prompt**. Credentials stay in your rclone
configuration on the cluster. See [rclone's headless setup guide](https://rclone.org/remote_setup/).

```bash
rclone mkdir gdrive:CASR_DAI2027
rclone lsd gdrive:CASR_DAI2027
```

Choose a dedicated Drive folder for each study. The launcher checks the saved
experiment plan and refuses to combine different code or scientific settings.

## Check the installation and recovery path

```bash
python run_experiments.py --self-test
python run_experiments.py --smoke --gpus 0,1 --output outputs/smoke01 --drive-remote gdrive:CASR_DAI2027/smoke01
```

The offline checks use tiny random models and rclone's local backend. They test
LoRA trainability, gradient routing, accumulation, atomic rejection, exact
checkpoint rollback, interrupted/resumed training, and versioned backup recovery.

The smoke run loads the real model on two GPUs, exercises native and LoRA
backward passes, writes Drive backups, and evaluates a very small subset. It
uses **scripted +1 advantages** to guarantee that an optimization step is
exercised even if a short test samples no correct answers. Smoke scores are
diagnostics and **must not enter paper results**. Inspect the logs and the
recorded `peak_cuda_allocated_gib` in `outputs/smoke01/summary.json`.

## Run all 24 paper fits

Keep the launcher alive across SSH disconnections with tmux:

```bash
tmux new -s casr-paper
source .venv/bin/activate
python run_experiments.py --gpus 0,1,2,3,4,5 --output outputs/paper01 --drive-remote gdrive:CASR_DAI2027/paper01
```

Detach with **Ctrl-B, then D**. Reconnect with:

```bash
tmux attach -t casr-paper
```

Each GPU runs one complete fit at a time. No NVLink, DDP, or model sharding is
needed. Device assignments follow a fixed shuffled job order. A failed fit is
recorded while other jobs continue; the launcher exits unsuccessfully if any
fit remains incomplete. Logs show the error to fix before resuming.

## Resume after a machine or session failure

Repeat the **same full-run command**, using the same code checkout, settings,
and Drive folder. Completed fits are skipped. If local storage disappeared,
the runner restores committed checkpoints and evidence from Drive. The
checkpoint contains both slots, all four Adam states, random-number generator
states, training history, and the development-selected checkpoint. The pinned
base model is downloaded again if its local cache was lost.

Drive backup is required. Every 20 attempted updates, and at development
boundaries, a consistent checkpoint is uploaded under a content hash. Its
remote MD5 is verified before a commit record is published. SHA-256 is checked
again on download. An interrupted upload cannot replace the previous restore
point. The two newest committed copies per fit are retained; cleanup is
restricted to checkpoints created by this run. Logs and evaluation caches are
copied without deleting destination files. If a backup fails, that worker stops
with its local checkpoint intact.

A machine failure may require replaying the updates after the last completed
backup (normally fewer than 20 updates). tmux protects against an SSH drop;
Drive protects against loss of the machine or local disk. Keep roughly
**6–10 GiB of free Drive storage**, plus room for logs/evaluation files, for the
24-fit study with two retained checkpoints per fit. There is no measured
RTX 2080 wall-time estimate yet.

## Outputs

* `outputs/paper01/final_results.csv`: final-step counts, including owner/all-on
  accuracy, foreign-only accuracy, and paired lost/rescued/changed answers.
* `outputs/paper01/summary.json`: completed fits, per-depth metrics, runtime,
  peak allocated GPU memory, and all-off integrity audits.
* `outputs/paper01/logs/`: one append-only log per fit.
* `outputs/paper01/fits/`: source/data manifests, raw predictions, evidence ZIPs,
  and local continuation bundles. The same evidence is backed up to Drive.

Results from this new FP32 study must be identified separately from the
earlier BF16 V6 results. The study tests bounded two-task post-training and
coexistence; it does not establish within-prompt skill composition, independent
merging, general reasoning localization, or a broad safety guarantee.

The V6 task/model/training source was carried forward from the project's
replication notebook. `dai_model.py`, `dai_rl.py`, `dai_drive.py`, and the CLI
add the LoRA comparator, memory-bounded accumulation, and cluster recovery.
