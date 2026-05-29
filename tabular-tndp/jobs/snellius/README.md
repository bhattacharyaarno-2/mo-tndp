# Snellius training

This folder contains Snellius/SLURM helpers for running the two long training jobs:

- `qlearning`: `train_qlearning.py`, CPU by default.
- `deep_rl`: `train_deep_rl.py`, GPU by default, using `gpu_mig` unless overridden.

Do not paste passwords or SURF tokens into these files. Use SSH, `ssh-agent`, and W&B login on the cluster.

## 1. Log in and inspect the account

From your laptop:

```bash
ssh <username>@snellius.surf.nl
```

If this times out, try SURF's doornode. Direct Snellius SSH is restricted to whitelisted IP ranges, while the doornode is reachable from anywhere:

```bash
ssh <username>@doornode.hpcv.surf.nl
```

After logging into the doornode, select `Snellius` from the menu and enter your password again. The doornode is only a login bridge; use direct Snellius SSH, a whitelisted network, VPN, or a remote Git clone for file transfer work.

On Snellius:

```bash
accinfo
accinfo --product=cpu
accinfo --product=gpu
accuse
budget-overview
sinfo -o "%P %a %l %D %C %G"
```

`accinfo` shows account validity and remaining budget. `budget-overview` is better once jobs are running because it accounts for recent and active jobs.

## 2. Upload the repo

From this repo on your laptop:

```bash
rsync -av --exclude .git --exclude .venv --exclude wandb --exclude logs --exclude q_tables --exclude deep_models ./ <username>@snellius.surf.nl:~/mo-tndp/
```

If you want to resume from local outputs, remove the relevant excludes.

## 3. Create the Python environment

On Snellius:

```bash
cd ~/mo-tndp
bash tabular-tndp/jobs/snellius/setup_env.sh
```

The setup script installs the CUDA PyTorch wheel from `https://download.pytorch.org/whl/cu126`, following the official PyTorch install-index pattern. Override it only if Snellius needs a different CUDA wheel:

```bash
PYTORCH_INDEX_URL='https://download.pytorch.org/whl/cu128' bash tabular-tndp/jobs/snellius/setup_env.sh
```

If the cluster module names differ, run `module avail 2>&1 | grep -Ei "mamba|conda|miniforge|python"` and edit the module section in `setup_env.sh`.

If you see an Lmod message saying `Miniconda3`, `Anaconda3`, or `Mamba` exists but cannot be loaded as requested, ask Lmod for the exact prerequisite/module path:

```bash
module spider Miniconda3
module spider Anaconda3
module spider Mamba
```

Then retry with the exact module name it prints:

```bash
CONDA_MODULE_STACK='2025' CONDA_MODULE='Miniconda3/<exact-version-or-path>' bash tabular-tndp/jobs/snellius/setup_env.sh
```

Use the same `CONDA_MODULE=...` prefix when submitting jobs if the batch script cannot auto-load conda.

For online W&B logging:

```bash
wandb login
```

For offline logging, submit with `--wandb-mode offline`.

## 4. Smoke test before burning budget

```bash
bash tabular-tndp/jobs/snellius/submit_sweep.sh qlearning amsterdam 2 42 1
bash tabular-tndp/jobs/snellius/submit_sweep.sh deep_rl amsterdam 2 42 1
```

Watch:

```bash
squeue -u "$USER"
tail -f logs/slurm-<jobid>_0.out
```

## 5. Submit serious sweeps

Ten seeds, 25k tabular episodes:

```bash
bash tabular-tndp/jobs/snellius/submit_sweep.sh qlearning amsterdam 25000 42 10
```

Ten seeds, 25k deep RL episodes:

```bash
bash tabular-tndp/jobs/snellius/submit_sweep.sh deep_rl amsterdam 25000 42 10
```

Useful environment overrides:

```bash
PARTITION=rome TIME=120:00:00 bash tabular-tndp/jobs/snellius/submit_sweep.sh qlearning amsterdam 100000 42 10
PARTITION=gpu_a100 TIME=120:00:00 bash tabular-tndp/jobs/snellius/submit_sweep.sh deep_rl amsterdam 100000 42 10
CPUS_PER_TASK=8 GPUS_PER_NODE=1 bash tabular-tndp/jobs/snellius/submit_sweep.sh deep_rl amsterdam 100000 42 10
WANDB_MODE=offline bash tabular-tndp/jobs/snellius/submit_sweep.sh deep_rl amsterdam 100000 42 10
```

## Cost sanity

Smallest Snellius allocations are charged in chunks. Rough guide:

- `rome`: smallest CPU allocation is 16 SBU/hour.
- `gpu_mig`: smallest GPU allocation is 64 SBU/hour.
- `gpu_a100`: smallest GPU allocation is 128 SBU/hour.
- `gpu_h100`: smallest GPU allocation is 192 SBU/hour.

So first measure episode throughput from the smoke-test logs, then scale the episode count and partition.
