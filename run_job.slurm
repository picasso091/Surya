#!/usr/bin/env bash
#SBATCH --account=bioq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=surya
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rakshya.panta@tcu.edu

set -euo pipefail
cd /u/rpanta2/Surya
mkdir -p logs/slurm

source .venv/bin/activate
export PYTHONUNBUFFERED=1

# ---- put the actual work here ----
.venv/bin/python -u downstream_examples/solar_flare_forcasting/experiments/cache_teacher_predictions.py \
  --manifest-dir downstream_examples/solar_flare_forcasting/experiments/manifests/pilot50_20260919T220418_102479Z
sbatch run_job.slurm
squeue --me          # check status
tail -F logs/slurm/surya-$(squeue --me -h -o %i | head -1).out

