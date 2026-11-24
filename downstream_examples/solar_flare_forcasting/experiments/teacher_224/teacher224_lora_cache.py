"""Teacher cache: the finetuned teacher at 224 on every train.csv sample.
Saves the logits (teacher_cache_train.csv) and the 1280-number CLS feature that enters the
classification layer (teacher_features_train.npy; row i belongs to row i of the CSV).

The teacher is the top trial in rank_trials.py's table, so run rank_trials.py first.
Run from downstream_examples/solar_flare_forcasting:
    CUDA_VISIBLE_DEVICES=0 conda run -n suryaenv --no-capture-output python experiments/teacher_224/teacher224_lora_cache.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from teacher224_lora import Flare224Dataset, add_lora, load_teacher_224


def main():
    cfg = yaml.safe_load(open("config_224.yaml"))
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="experiments/results_224/sweep_correct_css",
                   help="rank_trials.py --sweep_dir")
    p.add_argument("--selection", default="experiments/results_224/sweep_correct_css_selection.csv",
                   help="rank_trials.py --out")
    args = p.parse_args()
    d = cfg["data"]

    # 0. Teacher: top row of the selection table
    winner = pd.read_csv(args.selection).iloc[0]
    out = Path(args.sweep_dir) / winner["trial"]
    if not (out / "best.pth").exists():
        raise FileNotFoundError(f"{out}/best.pth not found: is {args.selection} from {args.sweep_dir}?")
    print(f"teacher: {out}  (rank_total {winner['rank_total']}, smoothed CSS {winner['smoothed_css']:.4f})")

    # 1. Training data, in file order
    train = Flare224Dataset(d, d["train_flare_data_path"])
    loader = DataLoader(train, batch_size=d["batch_size"], num_workers=d["num_data_workers"])

    # 2. The finetuned model with the weights from best.pth
    model = add_lora(load_teacher_224(cfg), cfg)
    model.load_state_dict(torch.load(out / "best.pth", map_location="cpu"))
    model = model.cuda().eval()

    # 3. Raw output (logit) for every sample, and the CLS feature the classification layer reads
    logits, features = [], []
    model.base_model.model.unembed.register_forward_hook(lambda layer, inp, out: features.append(inp[0].cpu()))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            logits.append(model(batch).cpu())
            if i % 200 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)
    logits = torch.cat(logits)

    # 4. Save one row per sample: logit, probability = sigmoid(logit), and the teacher's yes/no
    probs = torch.sigmoid(logits).numpy()
    threshold = json.load(open(out / "results.json"))["threshold"]
    cache = train.samples[["timestamp", "sample_id", d["label_column"]]].assign(
        teacher_logit=logits.numpy(), teacher_probability=probs, flare=probs >= threshold)
    cache.to_csv(out / "teacher_cache_train.csv", index=False)
    print(f"saved {out}/teacher_cache_train.csv ({len(cache)} samples)")
    features = torch.cat(features).numpy()
    np.save(out / "teacher_features_train.npy", features)
    print(f"saved {out}/teacher_features_train.npy {features.shape}")
    print(f"next: set student.teacher_folder in config_224.yaml to {out}")


if __name__ == "__main__":
    main()
