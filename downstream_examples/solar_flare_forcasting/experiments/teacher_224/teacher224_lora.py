"""Teacher at 224, finetuned with LoRA. Defaults come from config_224.yaml (the best sweep_css trial); flags override them.
CUDA_VISIBLE_DEVICES=1 conda run -n suryaenv --no-capture-output python experiments/teacher_224/teacher224_lora.py \
    --run_name lora_css
--sweep: one trial of sweep_css.yaml.
"""

import argparse
import json
import math
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
import zarr
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from surya.datasets.helio import transform
from surya.utils.data import build_scalers

from models import HelioSpectformer1D


class Flare224Dataset(Dataset):
    """(t-1h, t) image pair from the 224 zarr store, and the flare label at t."""

    def __init__(self, data_cfg, labels_csv):
        self.samples = pd.read_csv(labels_csv).merge(pd.read_csv(data_cfg["pairs_path"]), on="timestamp")
        self.zarr_path = data_cfg["zarr_path"]
        self.labels = self.samples[data_cfg["label_column"]].to_numpy()
        scalers = build_scalers(yaml.safe_load(open(data_cfg["scalers_path"])))
        self.stats = [np.array([getattr(scalers[c], a) for c in data_cfg["channels"]], dtype=np.float32)
                      for a in ("mean", "std", "sl_scale_factor", "epsilon")]
        print(f"{labels_csv}: {len(self.samples)} samples, {self.labels.mean():.1%} flares")

    def __len__(self):
        return len(self.samples)

    def frame(self, year, row):
        """One (13, 224, 224) image, normalized like the 4096 pipeline."""
        image = np.asarray(zarr.open(f"{self.zarr_path}/{year}/dataset/images", mode="r")[row], dtype=np.float32)
        return np.nan_to_num(transform(image, *self.stats)).astype(np.float32)

    def __getitem__(self, i):
        s = self.samples.iloc[i]
        ts = np.stack([self.frame(s.prev_year, s.prev_row), self.frame(s.curr_year, s.curr_row)], axis=1)
        return {"ts": torch.from_numpy(ts),                      # (13 channels, 2 times, 224, 224)
                "time_delta_input": torch.tensor([1.0, 0.0]),    # hours before t of each frame
                "label": torch.tensor(float(self.labels[i]))}


def crop_filter(weight, h, w):
    """Crop a spectral filter from the 4096 token grid (256x256) to the 224 grid (14x14), keeping the frequencies 14x14 has.
    weight: (256, 129, 1280, 2) -> (14, 8, 1280, 2)"""
    top = weight[: (h + 1) // 2, :w]                   # row frequencies 0..+6
    bottom = weight[weight.shape[0] - h // 2 :, :w]    # row frequencies -7..-1
    return torch.cat([top, bottom])


def load_teacher_224(cfg):
    """The 4096 teacher at 224: spectral filters cropped, pos_embed rebuilt, its LoRA merged in."""
    m = cfg["model"]
    model = HelioSpectformer1D(
        img_size=m["img_size"], patch_size=m["patch_size"], in_chans=m["in_channels"],
        embed_dim=m["embed_dim"], time_embedding=m["time_embedding"], depth=m["depth"],
        n_spectral_blocks=m["spectral_blocks"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
        drop_rate=m["drop_rate"], window_size=m["window_size"], dp_rank=m["dp_rank"],
        nglo=m["nglo"], dropout=m["dropout"], finetune=True, dtype=torch.float32,
        num_outputs=1, num_penultimate_transformer_layers=0, num_penultimate_heads=0, config=cfg,
    )
    # Same LoRA wrapping as the teacher's training, so the weight names match
    t = cfg["teacher_lora"]
    model = get_peft_model(model, LoraConfig(r=t["r"], lora_alpha=t["lora_alpha"], target_modules=t["target_modules"]))

    # pos_embed is skipped (the model built one for 196 tokens); any other weight not loading is an error
    teacher = torch.load(cfg["teacher_path"], map_location="cpu", weights_only=False)
    teacher.pop("base_model.model.embedding.pos_embed")
    state = model.state_dict()
    for name, weight in teacher.items():
        if name.endswith("complex_weight"):
            teacher[name] = crop_filter(weight, *state[name].shape[:2])
    missing, unexpected = model.load_state_dict(teacher, strict=False)
    assert missing == ["base_model.model.embedding.pos_embed"] and not unexpected, (missing, unexpected)

    return model.merge_and_unload()   # teacher's LoRA folded into fc1/fc2: a plain model again


def predict_logits(model, loader):
    """Logit for every sample, in dataset order."""
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            logits.append(model(batch).cpu())
    return torch.cat(logits)


def predict(model, loader):
    return torch.sigmoid(predict_logits(model, loader)).numpy()


def skill(labels, flare):
    """TSS, HSS, CSS and F1; flare: True where a flare is forecast."""
    tp = np.sum(flare & (labels == 1))
    fn = np.sum(~flare & (labels == 1))
    fp = np.sum(flare & (labels == 0))
    tn = np.sum(~flare & (labels == 0))
    return {"tss": float(tp / (tp + fn) - fp / (fp + tn)),
            "hss": float(2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))),
            "css": float(tp / (tp + fp) - fn / (fn + tn)),
            "f1": float(2 * tp / (2 * tp + fp + fn))}


def roc_auc(labels, probs):
    y = labels[np.argsort(-probs)]                                       # most flare-like first
    tpr, fpr = np.cumsum(y) / y.sum(), np.cumsum(1 - y) / (1 - y).sum()  # at each cut-off
    return float(np.sum(np.diff(np.r_[0, fpr]) * (np.r_[0, tpr][1:] + np.r_[0, tpr][:-1]) / 2))


def scores_at(labels, probs, cutoff):
    """Scores with a flare forecast where probs >= cutoff; forecast: how many flare forecasts.
    A CSS term that is 0/0 (no "yes" or no "no" forecasts) counts as 0."""
    flare = probs >= cutoff
    tp, fp = np.sum(flare & (labels == 1)), np.sum(flare & (labels == 0))
    fn, tn = np.sum(~flare & (labels == 1)), np.sum(~flare & (labels == 0))
    with np.errstate(divide="ignore", invalid="ignore"):   # skill's CSS may be 0/0; recomputed below
        s = skill(labels, flare)
    s["css"] = float((tp / (tp + fp) if tp + fp else 0.0) - (fn / (fn + tn) if fn + tn else 0.0))
    return {"roc_auc": roc_auc(labels, probs), "threshold": float(cutoff), "forecast": int(flare.sum()), **s} #here2


# def scores(labels, probs, by="css"):
#     """Scores at the cut-off with the highest `by` ("tss", "hss" or "css")."""
#     order = np.argsort(-probs)               # most flare-like first
#     y, p = labels[order], probs[order]
#     tp, fp = np.cumsum(y), np.cumsum(1 - y)  # with the cut-off at each sample
#     tpr, fpr = tp / tp[-1], fp / fp[-1]
#     tp, fp = tp.astype(float), fp.astype(float)
#     fn, tn = tp[-1] - tp, fp[-1] - fp
#     hss = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
#     tss = tpr - fpr
#     # tp + fp >= 1 at every cut-off; fn + tn is 0 at the last one, where that term counts as 0 (as in scores_at)
#     css = tp / (tp + fp) - np.divide(fn, fn + tn, out=np.zeros_like(fn), where=fn + tn > 0)
#     threshold = p[np.argmax({"tss": tss, "hss": hss, "css": css}[by])]
#     return {"roc_auc": float(np.sum(np.diff(np.r_[0, fpr]) * (np.r_[0, tpr][1:] + np.r_[0, tpr][:-1]) / 2)),
#             "threshold": float(threshold), **skill(labels, probs >= threshold)}


def round4(values):
    """Scores to 4 decimals. Thresholds stay exact: later scripts reapply them, and rounding could flip samples."""
    return {k: round(float(v), 4) if isinstance(v, float) and "threshold" not in k else v for k, v in values.items()}


def show(values):
    return " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in values.items())


def stop_early(history, patience, metric="roc_auc"):
    """True once `metric` hasn't improved for `patience` checks; patience None never stops."""
    best = max(range(len(history)), key=lambda i: history[i][metric])
    return patience is not None and len(history) - 1 - best >= patience


def add_lora(model, cfg):
    """Trainable: new LoRA on cfg["lora"] layers, and the whole patch embedding."""
    model = get_peft_model(model, LoraConfig(**cfg["lora"]))  # freezes every original weight
    for name, p in model.named_parameters():
        if "patch_embed" in name:
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable / 1e6:.2f} M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    return model


def lr_schedule(optimizer, decay, total_steps, warmup):
    """Multiplier of learning_rate per step. none: 1; cosine: 1 -> 0 along half a cosine;
    warmup_cosine: 0 -> 1 linearly over the first `warmup` share of steps, then cosine."""
    w = max(1, round(warmup * total_steps))
    multiple = {"none": lambda s: 1.0,
                "cosine": lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)),
                "warmup_cosine": lambda s: (s + 1) / w if s < w
                else 0.5 * (1 + math.cos(math.pi * (s - w) / (total_steps - w)))}[decay]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiple)


def main():
    
    # 0. Reproducible runs (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before the GPU is first used

    cfg = yaml.safe_load(open("config_224.yaml"))
    d, tr = cfg["data"], cfg["train"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=cfg["run_name"], help="results subfolder and wandb run name")
    parser.add_argument("--epochs", type=int, default=tr["epochs"])
    parser.add_argument("--patience", type=lambda s: None if s == "none" else int(s),
                        default=tr.get("early_stopping_patience"),
                        help="checks without a better validation CSS before stopping; none trains to the end")
    parser.add_argument("--learning_rate", type=float, default=tr["learning_rate"]) #here1
    parser.add_argument("--weight_decay", type=float, default=tr["weight_decay"])
    parser.add_argument("--decay", choices=["none", "cosine", "warmup_cosine"], default=tr["decay"],
                        help="learning-rate schedule (lr_schedule)")
    parser.add_argument("--seed", type=int, default=0, help="model and shuffling randomness")
    parser.add_argument("--sweep", action="store_true",
                        help="a search trial: its own folder, <run_name>/<decay>_lr<..>_wd<..>_<start time>")
    args = parser.parse_args()
    for k in ("epochs", "learning_rate", "weight_decay", "decay", "seed"):   # into tr, so wandb records them
        tr[k] = getattr(args, k)
    tr["early_stopping_patience"] = args.patience
    cfg["run_name"] = (f"{args.run_name}/{args.decay}_lr{args.learning_rate:.1e}_wd{args.weight_decay:.1e}_"
                       f"{time.strftime('%m%d-%H%M%S')}" if args.sweep else args.run_name)
    out = Path(cfg["output_folder"]) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # 1. Data
    train = Flare224Dataset(d, d["train_flare_data_path"])
    train_loader = DataLoader(train, shuffle=True, batch_size=d["batch_size"],
                              num_workers=d["num_data_workers"], pin_memory=True)
    valid = Flare224Dataset(d, d["valid_flare_data_path"])
    valid_loader = DataLoader(valid, batch_size=d["batch_size"], num_workers=d["num_data_workers"])
    # Fixed train subset, validation-sized and scored like validation, to spot overfitting
    train_subset = np.random.default_rng(0).choice(len(train), len(valid), replace=False)
    train_subset_loader = DataLoader(Subset(train, train_subset), batch_size=d["batch_size"],
                                     num_workers=d["num_data_workers"])

    # 2. Model
    model = add_lora(load_teacher_224(cfg), cfg).cuda()

    # 3. Optimizer and learning-rate schedule
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tr["learning_rate"], weight_decay=tr["weight_decay"]) #here

    total_steps = tr["epochs"] * len(train_loader)
    scheduler = lr_schedule(optimizer, tr["decay"], total_steps, tr.get("warmup", 0.05))
    
    
    
    # Weighted BCE: each flare weighs non-flares/flares, balancing the classes
    flare_rate = train.labels.mean()
    pos_weight = torch.tensor(float((1 - flare_rate) / flare_rate) if tr["weighted_bce"] else 1.0, device="cuda")
    print(f"flare sample weight in the loss: {pos_weight.item():.2f}")

    run = wandb.init(project=cfg["wandb"]["project"], name=cfg["run_name"].replace("/", "-"), config=cfg,
                     mode=cfg["wandb"]["mode"]) if cfg.get("wandb") else None

    # 4. Each check saves best.pth on a new best validation CSS and returns True to stop early.
    #    Step 0 (the teacher before finetuning) is logged but never counted as best.
    history, losses = [], []
    patience = tr["early_stopping_patience"]

    def measure(loader, labels):
        """Weighted-BCE loss and scores, dropout off."""
        logits = predict_logits(model, loader)
        loss = F.binary_cross_entropy_with_logits(logits, torch.tensor(labels, dtype=torch.float32), pos_weight=pos_weight.cpu())
        return loss.item(), scores_at(labels, torch.sigmoid(logits).numpy(), tr["cutoff"])

    def check(step):
        valid_loss, v = measure(valid_loader, valid.labels)
        subset_loss, t = measure(train_subset_loader, train.labels[train_subset])
        is_best = step > 0 and (len(history) == 1 or v["css"] > max(h["css"] for h in history[1:]))
        history.append({"step": step,
                        "train_loss": float(np.mean(losses[-tr["validate_every"]:] or [np.nan])),  # running mean, dropout on
                        "train_subset_loss": subset_loss, "valid_loss": valid_loss,
                        "train_subset_roc_auc": t["roc_auc"], "train_subset_css": t["css"], **v})
        print(show(history[-1]), flush=True)
        if run:   # valid/best_css: the sweep's metric
            run.log({**{f"valid/{k}": x for k, x in v.items()}, "valid/loss": valid_loss,
                     **{f"train_subset/{k}": x for k, x in t.items()}, "train_subset/loss": subset_loss,
                     **({"valid/best_css": max(h["css"] for h in history[1:])} if step > 0 else {})}, step=step)
        if is_best:
            torch.save(model.state_dict(), out / "best.pth")
        return step > 0 and stop_early(history[1:], patience, "css")

    # 5. Train, checking every validate_every steps and at the last step
    check(0)
    step, stopped = 0, False
    for epoch in range(tr["epochs"]):
        if stopped:
            break
        for batch in train_loader:
            model.train()
            batch = {k: x.cuda() for k, x in batch.items()}
            loss = F.binary_cross_entropy_with_logits(model(batch), batch["label"], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if run:
                run.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)
            if (step % tr["validate_every"] == 0 or step == total_steps) and check(step):
                print(f"early stop at step {step}: no better validation CSS in the last {patience} checks")
                stopped = True
                break

    # 6. Score the best checkpoint on validation
    model.load_state_dict(torch.load(out / "best.pth", map_location="cpu"))
    probs = predict(model, valid_loader)
    result = round4({"best_step": int(max(history[1:], key=lambda h: h["css"])["step"]), "last_step": step,
                     "pos_weight": pos_weight.item(), **scores_at(valid.labels, probs, tr["cutoff"])})
    print(json.dumps(result, indent=1))

    # 7. Save
    json.dump(result, open(out / "results.json", "w"), indent=1)
    pd.DataFrame([round4(h) for h in history]).to_csv(out / "history.csv", index=False)
    valid.samples.assign(prob=probs, flare=probs >= result["threshold"])[
        ["timestamp", d["label_column"], "prob", "flare"]].to_csv(out / "valid_predictions.csv", index=False)
    print(f"saved {out}/results.json, history.csv and valid_predictions.csv")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
