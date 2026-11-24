"""Create a wandb Bayesian sweep for one student and one distillation method (distillation-plan.txt).
Search space: the teacher's (teacher_224/sweep_css.yaml) plus the teacher weight beta. Each trial runs
train_student224.py; the settings not searched (epochs, patience, ...) come from the `student` section of config_224.yaml.

From downstream_examples/solar_flare_forcasting:
    /home/rakshyapanta/.conda/envs/suryaenv/bin/python experiments/student/sweep_student224.py --model cnn --kd logit
then start an agent per GPU with the `wandb agent <sweep id>` line it prints:
    CUDA_VISIBLE_DEVICES=0 /home/rakshyapanta/.conda/envs/suryaenv/bin/wandb agent --count 20 <sweep id>

--kd none: the labels-only sweep (no beta), the plan's second baseline.
"""

import argparse
import sys

import wandb
import yaml

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["cnn", "tinyvit"], required=True)
p.add_argument("--kd", choices=["none", "logit", "feature"], required=True)
p.add_argument("--trials", type=int, default=20)
args = p.parse_args()

parameters = {
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-6, "max": 3e-4},
    "weight_decay": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
    "decay": {"values": ["none", "cosine", "warmup_cosine"]},   # none = constant learning rate
    "pos_weight": {"values": ["plain", "half", "full"]},        # class weight; plain = no class weight
}
if args.kd != "none":
    parameters["beta"] = {"values": [0, 0.5, 1, 2]}             # 0 = labels only

wandb.sweep({
    "name": f"{args.model}_{args.kd}",
    "program": "experiments/student/train_student224.py",
    "method": "bayes",
    "metric": {"name": "valid/best_css", "goal": "maximize"},
    "run_cap": args.trials,
    "parameters": parameters,
    "command": [sys.executable, "${program}", "--model", args.model, "--kd", args.kd, "${args}"],
}, project=yaml.safe_load(open("config_224.yaml"))["wandb"]["project"])
