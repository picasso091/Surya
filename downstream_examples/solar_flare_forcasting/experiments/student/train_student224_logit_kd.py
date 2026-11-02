"""Student at 224 with logit KD: 
    alpha * weighted BCE(labels) + beta * MSE(cached teacher logits).
    
--beta 0: the supervised baseline.
--sweep: one trial of sweep_student_sup.yaml / sweep_student_kd.yaml

Run from downstream_examples/solar_flare_forcasting:
CUDA_VISIBLE_DEVICES=1 conda run -n suryaenv --no-capture-output python experiments/student/train_student224_logit_kd.py \
    --beta 0.5 --output_folder experiments/results_224/mbn_logit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teacher_224"))

from teacher224_lora import Flare224Dataset, lr_schedule, round4, scores_at, show, stop_early  # same data and metrics as the teacher
from student224_cnn import StudentCNN, logit_kd_loss
from student224_mobilenet import StudentMobileNet

STUDENTS = {"mobilenet": StudentMobileNet, "cnn": StudentCNN}


class DistillDataset(Flare224Dataset):
    """The teacher's dataset, plus each sample's cached teacher logit."""

    def __init__(self, data_cfg, labels_csv, cache_csv):
        super().__init__(data_cfg, labels_csv)
        n = len(self.samples)
        cache = pd.read_csv(cache_csv)[["timestamp", "teacher_logit"]]
        self.samples = self.samples.merge(cache, on="timestamp", validate="one_to_one")
        assert len(self.samples) == n, f"{n - len(self.samples)} training samples have no teacher logit"
        self.labels = self.samples[data_cfg["label_column"]].to_numpy()

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["teacher_logit"] = torch.tensor(float(self.samples["teacher_logit"].iloc[i]))
        return item


def predict(model, loader):
    """Flare probability for every sample, in dataset order."""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            probs.append(torch.sigmoid(model(batch["ts"].cuda())).squeeze(1).cpu())
    return torch.cat(probs).numpy()


def save_predictions(dataset, probs, threshold, path, label_column):
    dataset.samples.assign(prob=probs, flare=probs >= threshold)[
        ["timestamp", label_column, "prob", "flare"]].to_csv(path, index=False)


def parse_args():
    """Overrides of the `student` config section; a flag left out keeps the config value."""
    p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    p.add_argument("--beta", type=float, help="weight of the teacher term; 0 = no distillation")
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--decay", choices=["none", "cosine", "warmup_cosine"], help="learning-rate schedule (lr_schedule)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", dest="early_stopping_patience", type=lambda x: None if x == "none" else int(x),
                   help="checks without a better `select` before stopping; none trains to the end")
    p.add_argument("--select", choices=["roc_auc", "css"], help="validation score that picks best.pth")
    p.add_argument("--seed", type=int, help="student initialization and data order (default 0)")
    p.add_argument("--model", choices=list(STUDENTS), help="default: config, else mobilenet")
    p.add_argument("--output_folder")
    p.add_argument("--sweep", action="store_true", help="a search trial: own subfolder, no test scoring")
    return vars(p.parse_args())


def main():
    # 0. Reproducible runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    cfg = yaml.safe_load(open("config_224.yaml"))
    d, s = cfg["data"], cfg["student"]
    args = parse_args()
    sweep = args.pop("sweep", False)
    s.update(args)  # flags override the config
    s.setdefault("seed", 0)
    s.setdefault("model", "mobilenet")
    metric = s["select"]
    cutoff = cfg["train"]["cutoff"]  # the teacher's fixed cut-off
    out = Path(s["output_folder"])
    if sweep:
        out /= (f"{s['decay']}_lr{s['learning_rate']:.1e}_wd{s['weight_decay']:.1e}_b{s['beta']:.2g}_"
                f"{time.strftime('%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(s["seed"])  # student initialization and training-data order
    make_loader = lambda dataset, shuffle=False: DataLoader(dataset, batch_size=d["batch_size"], shuffle=shuffle,
                                                            num_workers=d["num_data_workers"], pin_memory=True)

    # 1. Data
    train = DistillDataset(d, d["train_flare_data_path"], s["teacher_cache"])
    valid = Flare224Dataset(d, d["valid_flare_data_path"])
    train_loader, valid_loader = make_loader(train, shuffle=True), make_loader(valid)

    # 2. Student, optimizer and learning-rate schedule
    model = STUDENTS[s["model"]]().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
    total_steps = s["epochs"] * len(train_loader)
    scheduler = lr_schedule(optimizer, s["decay"], total_steps, s.get("warmup", 0.05))
    print(f"student {s['model']}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, {total_steps} steps")

    # 3. Weighted BCE: each flare weighs non-flares/flares, balancing the classes
    flare_rate = train.labels.mean()
    pos_weight = torch.tensor(float((1 - flare_rate) / flare_rate), device="cuda")
    print(f"flare sample weight {pos_weight.item():.2f}, alpha {s['alpha']}, beta {s['beta']}")
    run = wandb.init(project=cfg["wandb"]["project"], name="-".join(out.parts[-2:]) if sweep else out.name,
                     config=cfg, mode=cfg["wandb"]["mode"]) if cfg.get("wandb") else None

    # 4. Each check saves best.pth on a new best validation `metric` and returns True to stop early.
    #    Step 0 (the untrained student) is logged but never counted as best.
    history, parts = [], []
    patience = s.get("early_stopping_patience")

    def check(step, epoch):
        v = scores_at(valid.labels, predict(model, valid_loader), cutoff)
        is_best = step > 0 and (len(history) == 1 or v[metric] > max(h[metric] for h in history[1:]))
        recent = parts[-s["validate_every"]:]  # training losses since the last check
        history.append({"step": step, "epoch": epoch,
                        **{k: np.mean([p[k] for p in recent]) if recent else np.nan for k in ("loss", "wbce", "logit_mse")},
                        **v})
        print(show(history[-1]), flush=True)
        if run:   # valid/best_<metric>: the sweep's metric
            run.log({**{f"valid/{k}": x for k, x in v.items()},
                     **({f"valid/best_{metric}": max(h[metric] for h in history[1:])} if step > 0 else {})}, step=step)
        if is_best:
            torch.save(model.state_dict(), out / "best.pth")
        return step > 0 and stop_early(history[1:], patience, metric)

    # 5. Train, checking every validate_every steps and at the last step
    check(0, 0)
    step, stopped = 0, False
    for epoch in range(s["epochs"]):
        if stopped:
            break
        for batch in train_loader:
            model.train()
            logits = model(batch["ts"].cuda(non_blocking=True))
            loss, p = logit_kd_loss(logits, batch["teacher_logit"].cuda(), batch["label"].cuda(),
                                    pos_weight, s["alpha"], s["beta"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            parts.append({"loss": loss.item(), **p})
            step += 1
            if run:
                run.log({"train/loss": loss.item(), **{f"train/{k}": x for k, x in p.items()},
                         "lr": scheduler.get_last_lr()[0]}, step=step)
            if (step % s["validate_every"] == 0 or step == total_steps) and check(step, epoch):
                print(f"early stop at step {step}: no better validation {metric} in the last {patience} checks")
                stopped = True
                break

    # 6. Score the best checkpoint on validation
    model.load_state_dict(torch.load(out / "best.pth", map_location="cuda"))
    probs = predict(model, valid_loader)
    result = round4({"model": s["model"], "seed": s["seed"], "alpha": s["alpha"], "beta": s["beta"], "select": metric,
                     "best_step": int(max(history[1:], key=lambda h: h[metric])["step"]), "last_step": step,
                     "pos_weight": pos_weight.item(), **scores_at(valid.labels, probs, cutoff)})
    save_predictions(valid, probs, result["threshold"], out / "valid_predictions.csv", d["label_column"])
    json.dump(result, open(out / "results.json", "w"), indent=1)
    pd.DataFrame([round4(h) for h in history]).to_csv(out / "history.csv", index=False)
    print(f"validation: {json.dumps(result, indent=1)}")

    # 7. Test at the same cut-off; skipped in a sweep so trials are compared on validation only
    if not sweep:
        test = Flare224Dataset(d, d["test_flare_data_path"])
        test_probs = predict(model, make_loader(test))
        test_result = round4(scores_at(test.labels, test_probs, cutoff))
        save_predictions(test, test_probs, result["threshold"], out / "test_predictions.csv", d["label_column"])
        json.dump(test_result, open(out / "test_results.json", "w"), indent=1)
        print(f"test: {json.dumps(test_result, indent=1)}")
    print(f"saved in {out}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
