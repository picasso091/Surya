"""Train one student (CNN or TinyViT) at 224 with knowledge distillation from the cached teacher (distillation-plan.txt).

    loss = WBCE(student logit, label) + beta * distillation term          (alpha, the label weight, is fixed at 1)
    --kd logit:    distillation term = mean (student logit - teacher logit)^2
    --kd feature:  distillation term = mean (1 - cosine(g(student feature), teacher CLS feature))
    --kd none:     labels only, the supervised baseline (beta is ignored)
    beta 0 with logit or feature KD also trains on the labels only.

Settings come from the `student` section of config_224.yaml; flags override them (a sweep passes its trial's values).
Each run saves best.pth (best validation CSS at train.cutoff), results.json, history.csv and valid_predictions.csv into
experiments/results_224/students/<model>_<kd>/<run name>/. Test scores are separate: test_student224.py.

Run from downstream_examples/solar_flare_forcasting, after teacher224_lora_cache.py:
    CUDA_VISIBLE_DEVICES=1 conda run -n suryaenv --no-capture-output python experiments/student/train_student224.py \
        --model cnn --kd logit --beta 1
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
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teacher_224"))

from teacher224_lora import Flare224Dataset, lr_schedule, round4, scores_at, show, stop_early  # same data and metrics as the teacher
from student224_cnn import StudentCNN
from student224_tinyvit import StudentTinyViT

STUDENTS = {"cnn": StudentCNN, "tinyvit": StudentTinyViT}


class TeacherCacheDataset(Flare224Dataset):
    """Training samples plus the teacher's cached outputs (teacher224_lora_cache.py):
    teacher_logit from teacher_cache_train.csv, teacher_feature [1280] from teacher_features_train.npy (row i = CSV row i).
    """

    def __init__(self, data_cfg, teacher_folder):
        super().__init__(data_cfg, data_cfg["train_flare_data_path"])
        cache = pd.read_csv(Path(teacher_folder) / "teacher_cache_train.csv")
        cache["feature_row"] = np.arange(len(cache))
        n = len(self.samples)
        self.samples = self.samples.merge(cache[["sample_id", "teacher_logit", "feature_row"]], on="sample_id",
                                          validate="one_to_one")
        assert len(self.samples) == n, f"{n - len(self.samples)} training samples are missing from the teacher cache"
        self.labels = self.samples[data_cfg["label_column"]].to_numpy()
        self.features = np.load(Path(teacher_folder) / "teacher_features_train.npy", mmap_mode="r")  # [N, 1280]
        assert len(self.features) == len(cache), "teacher feature file and cache CSV have different lengths"

    def __getitem__(self, i):
        item = super().__getitem__(i)
        s = self.samples.iloc[i]
        item["teacher_logit"] = torch.tensor(float(s.teacher_logit))
        item["teacher_feature"] = torch.from_numpy(np.array(self.features[int(s.feature_row)], dtype=np.float32))
        return item


def student_loss(logit, feature, batch, kd, beta, pos_weight):
    """WBCE on the labels + beta * distillation term; also returns both terms, for logging."""
    wbce = F.binary_cross_entropy_with_logits(logit, batch["label"], pos_weight=pos_weight)
    if kd == "logit":
        distill = F.mse_loss(logit, batch["teacher_logit"])
    elif kd == "feature":
        distill = (1 - F.cosine_similarity(feature, batch["teacher_feature"], dim=1)).mean()
    else:
        distill = torch.zeros_like(wbce)
    return wbce + beta * distill, {"wbce": wbce.item(), "distill": distill.item()}


def predict(model, loader):
    """Flare probability for every sample, in dataset order."""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            logit, _ = model(batch["ts"].cuda())
            probs.append(torch.sigmoid(logit).squeeze(1).cpu())
    return torch.cat(probs).numpy()


def parse_args():
    """Overrides of the `student` config section; a flag left out keeps the config value."""
    p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    p.add_argument("--model", choices=list(STUDENTS))
    p.add_argument("--kd", choices=["none", "logit", "feature"], help="distillation method; none = labels only")
    p.add_argument("--beta", type=float, help="weight of the distillation term; 0 = labels only")
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--decay", choices=["none", "cosine", "warmup_cosine"], help="learning-rate schedule (lr_schedule)")
    p.add_argument("--pos_weight", choices=["plain", "half", "full"],
                   help="flare weight in the BCE: plain 1 (no class weight), half or full non-flares/flares")
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", dest="early_stopping_patience", type=lambda x: None if x == "none" else int(x),
                   help="checks without a better validation CSS before stopping; none trains to the end")
    p.add_argument("--seed", type=int, help="student initialization and training-data order")
    return vars(p.parse_args())


def main():
    # 0. Reproducible runs (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before the GPU is first used

    cfg = yaml.safe_load(open("config_224.yaml"))
    d, s = cfg["data"], cfg["student"]
    s.update(parse_args())  # flags override the config
    if s["kd"] == "none":
        s["beta"] = 0.0
    cutoff = cfg["train"]["cutoff"]  # the teacher's fixed cut-off
    group = f"{s['model']}_{s['kd']}"
    name = (f"b{s['beta']:g}_{s['decay']}_lr{s['learning_rate']:.1e}_wd{s['weight_decay']:.1e}_pw{s['pos_weight']}_"
            f"s{s['seed']}_{time.strftime('%m%d-%H%M%S')}")
    out = Path(cfg["output_folder"]) / "students" / group / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(s["seed"])  # student initialization and training-data order
    make_loader = lambda dataset, shuffle=False: DataLoader(dataset, batch_size=d["batch_size"], shuffle=shuffle,
                                                            num_workers=d["num_data_workers"], pin_memory=True)

    # 1. Data: training samples with the teacher's cached outputs; validation
    train = TeacherCacheDataset(d, s["teacher_folder"])
    valid = Flare224Dataset(d, d["valid_flare_data_path"])
    train_loader, valid_loader = make_loader(train, shuffle=True), make_loader(valid)

    # 2. Student, AdamW and learning-rate schedule
    model = STUDENTS[s["model"]]().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
    total_steps = s["epochs"] * len(train_loader)
    scheduler = lr_schedule(optimizer, s["decay"], total_steps, s["warmup"])
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"student {s['model']}: backbone {n(model.backbone) / 1e6:.2f} M, classifier {n(model.classifier)}, "
          f"feature head {n(model.feature_head) / 1e6:.2f} M; {total_steps} steps")

    # 3. Flare weight in the BCE, as for the teacher: full = non-flares/flares balances the classes
    flare_rate = train.labels.mean()
    balanced = float((1 - flare_rate) / flare_rate)
    pos_weight = torch.tensor({"plain": 1.0, "half": balanced / 2, "full": balanced}[s["pos_weight"]], device="cuda")
    print(f"kd {s['kd']}, beta {s['beta']}, flare weight {pos_weight.item():.2f} ({s['pos_weight']})")
    run = wandb.init(project=cfg["wandb"]["project"], name=f"{group}-{name}", group=group, config=cfg,
                     mode=cfg["wandb"]["mode"]) if cfg.get("wandb") else None

    # 4. Each check saves best.pth on a new best validation CSS and returns True to stop early.
    #    Step 0 (the untrained student) is logged but never counted as best.
    history, losses = [], []
    patience = s["early_stopping_patience"]

    def check(step, epoch):
        v = scores_at(valid.labels, predict(model, valid_loader), cutoff)
        is_best = step > 0 and v["css"] > max((h["css"] for h in history[1:]), default=-1.0)
        recent = losses[-s["validate_every"]:]  # training losses since the last check
        history.append({"step": step, "epoch": epoch,
                        **{k: np.mean([x[k] for x in recent]) if recent else np.nan for k in ("loss", "wbce", "distill")},
                        **v})
        print(show(history[-1]), flush=True)
        if run:  # valid/best_css: the sweep's metric
            run.log({**{f"valid/{k}": x for k, x in v.items()},
                     **({"valid/best_css": max(h["css"] for h in history[1:])} if step > 0 else {})}, step=step)
        if is_best:
            torch.save(model.state_dict(), out / "best.pth")
        return step > 0 and stop_early(history[1:], patience, "css")

    # 5. Train, checking every validate_every steps and at the last step
    check(0, 0)
    step, stopped = 0, False
    for epoch in range(s["epochs"]):
        if stopped:
            break
        for batch in train_loader:
            model.train()
            batch = {k: x.cuda(non_blocking=True) for k, x in batch.items()}
            logit, feature = model(batch["ts"])
            loss, parts = student_loss(logit.squeeze(1), feature, batch, s["kd"], s["beta"], pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append({"loss": loss.item(), **parts})
            step += 1
            if run:
                run.log({"train/loss": loss.item(), **{f"train/{k}": x for k, x in parts.items()},
                         "lr": scheduler.get_last_lr()[0]}, step=step)
            if (step % s["validate_every"] == 0 or step == total_steps) and check(step, epoch):
                print(f"early stop at step {step}: no better validation CSS in the last {patience} checks")
                stopped = True
                break

    # 6. Score the best checkpoint on validation; the settings stay unrounded (learning rates go down to 1e-6)
    model.load_state_dict(torch.load(out / "best.pth", map_location="cuda"))
    probs = predict(model, valid_loader)
    settings = {k: s[k] for k in ("model", "kd", "beta", "learning_rate", "weight_decay", "decay", "pos_weight",
                                  "seed", "epochs", "teacher_folder")}
    result = {**settings, **round4({"best_step": int(max(history[1:], key=lambda h: h["css"])["step"]),
                                    "last_step": step, "pos_weight_value": pos_weight.item(),
                                    **scores_at(valid.labels, probs, cutoff)})}
    print(f"validation: {json.dumps(result, indent=1)}")

    # 7. Save
    json.dump(result, open(out / "results.json", "w"), indent=1)
    pd.DataFrame([round4(h) for h in history]).to_csv(out / "history.csv", index=False)
    valid.samples.assign(prob=probs, flare=probs >= cutoff)[
        ["timestamp", d["label_column"], "prob", "flare"]].to_csv(out / "valid_predictions.csv", index=False)
    print(f"saved in {out}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
