"""Finetune the solar-flare model (teacher) on 224x224 images.

Run from downstream_examples/solar_flare_forcasting:
    CUDA_VISIBLE_DEVICES=0 python finetune_224.py --config config_224.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import yaml
import zarr
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from surya.datasets.helio import transform
from surya.utils.data import build_scalers

from finetune import apply_peft_lora
from metrics import DistributedClassificationMetrics
from models import HelioSpectformer1D


# Step 1. Data: (t-1h, t) image pair from the 224 zarr store, and the flare label at t
class Flare224Dataset(Dataset):
    def __init__(self, data_cfg, labels_csv, scalers):
        self.samples = pd.read_csv(labels_csv).merge(pd.read_csv(data_cfg["pairs_path"]), on="timestamp")
        self.zarr_path = data_cfg["zarr_path"]
        self.label_column = data_cfg["label_column"]
        self.stats = [np.array([getattr(scalers[c], a) for c in data_cfg["channels"]], dtype=np.float32)
                      for a in ("mean", "std", "sl_scale_factor", "epsilon")]
        print(f"{labels_csv}: {len(self.samples)} samples, {self.samples[self.label_column].mean():.1%} flares")

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
                "label": torch.tensor(float(s[self.label_column]))}


# Step 2. Model: the teacher's architecture at 224, with the teacher's weights
def crop_filter(weight, h, w):
    """Keep the lowest frequencies of a spectral filter (256x129 -> 14x8).
    Rows run 0, +1, ... and end ..., -2, -1, so low frequencies sit at both ends."""
    return torch.cat([weight[: (h + 1) // 2, :w], weight[weight.shape[0] - h // 2 :, :w]])


def build_model(cfg):
    m = cfg["model"]
    model = HelioSpectformer1D(
        img_size=m["img_size"], patch_size=m["patch_size"], in_chans=m["in_channels"],
        embed_dim=m["embed_dim"], time_embedding=m["time_embedding"], depth=m["depth"],
        n_spectral_blocks=m["spectral_blocks"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
        drop_rate=m["drop_rate"], window_size=m["window_size"], dp_rank=m["dp_rank"],
        nglo=m["nglo"], dropout=m["dropout"], finetune=True, dtype=torch.float32,
        num_outputs=1, num_penultimate_transformer_layers=0, num_penultimate_heads=0, config=cfg,
    )

    # 2a. Wrap model with LoRA like the teacher was trained (LoRA on fc1, fc2)
    t = cfg["teacher_lora"]
    model = get_peft_model(model, LoraConfig(r=t["r"], lora_alpha=t["lora_alpha"],
                                             target_modules=t["target_modules"]))

    # 2b. Load the teacher. Only two weights change size at 224: the spectral filters are cropped, pos_embed is rebuilt for 196 tokens. Other layers are not input dependent.
    teacher = torch.load(cfg["teacher_path"], map_location="cpu", weights_only=False)
    teacher.pop("base_model.model.embedding.pos_embed") #Removes the teacher's position pattern, which was sized for 65,536 tokens.
    state = model.state_dict()
    for name, weight in teacher.items():
        if name.endswith("complex_weight"):
            teacher[name] = crop_filter(weight, *state[name].shape[:2])
    # Copies every teacher tensor into the model entry and returns a list of missing and unexpected keys.
    missing, unexpected = model.load_state_dict(teacher, strict=False)
    assert missing == ["base_model.model.embedding.pos_embed"] and not unexpected, (missing, unexpected) # the only missing key is the pos_embed, which is rebuilt for 224x224. There should be no unexpected keys.
    print("Loaded the teacher: every weight, spectral filters cropped, pos_embed rebuilt")

    # 2c. Run A: use_lora:true, fold the teacher's LoRA into fc1/fc2, add new LoRA to the big layers, and train only that new LoRA + train_fully ,freeze big layers(attn.qkv,attn.proj, mlp.fc1,mlp.fc2). 
        # Run B: use_lora:false, train big layers + lora weights+ train_fully layers.
    if m["use_lora"]:
        model = apply_peft_lora(model.merge_and_unload(), cfg)
        for name, p in model.named_parameters():
            if any(s in name for s in m["train_fully"]):
                p.requires_grad = True
    else:
        model.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable / 1e6:.2f} M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    return model


# Step 3. One pass over the data: trains if an optimizer is given, otherwise only evaluates
def run_epoch(model, loader, optimizer=None, run=None):
    model.train(optimizer is not None)
    metrics = DistributedClassificationMetrics(threshold=0.5)
    losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"  batch {i}/{len(loader)}  loss {loss.item():.4f}", flush=True)
                    if run:
                        run.log({"train/batch_loss": loss.item()})
            metrics.update(torch.sigmoid(logits.detach()), batch["label"])  # logits -> probabilities
            losses.append(loss.item())
    return np.mean(losses), metrics.compute_and_reset()


# Step 4. Train, validate after every epoch, keep the checkpoint with the best validation TSS
def main(cfg):
    torch.manual_seed(0)
    d = cfg["data"]
    scalers = build_scalers(yaml.safe_load(open(d["scalers_path"])))
    train_loader = DataLoader(Flare224Dataset(d, d["train_flare_data_path"], scalers), shuffle=True,
                              batch_size=d["batch_size"], num_workers=d["num_data_workers"], pin_memory=True)
    valid_loader = DataLoader(Flare224Dataset(d, d["valid_flare_data_path"], scalers),
                              batch_size=d["batch_size"], num_workers=d["num_data_workers"], pin_memory=True)

    model = build_model(cfg).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    os.makedirs(cfg["path_experiment"], exist_ok=True)

    run = wandb.init(project=cfg["wandb"]["project"], name=cfg["path_experiment"], config=cfg,
                     mode=cfg["wandb"]["mode"]) if cfg.get("wandb") else None

    best_tss = -1.0
    for epoch in range(cfg["max_epochs"]):
        train_loss, train = run_epoch(model, train_loader, optimizer, run)
        valid_loss, valid = run_epoch(model, valid_loader)
        tss = float(valid["tss"])
        print(f"epoch {epoch}: train loss {train_loss:.4f} | valid loss {valid_loss:.4f} "
              f"tss {tss:.3f} f1 {float(valid['f1']):.3f}", flush=True)
        if run:
            run.log({"epoch": epoch, "train/loss": train_loss, "valid/loss": valid_loss,
                     **{f"train/{k}": float(v) for k, v in train.items()},
                     **{f"valid/{k}": float(v) for k, v in valid.items()}})
        torch.save(model.state_dict(), f"{cfg['path_experiment']}/epoch_{epoch}.pth")
        if tss > best_tss:
            best_tss = tss
            torch.save(model.state_dict(), f"{cfg['path_experiment']}/best_tss.pth")
            print(f"  new best validation TSS {best_tss:.3f}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_224.yaml")
    main(yaml.safe_load(open(parser.parse_args().config)))
