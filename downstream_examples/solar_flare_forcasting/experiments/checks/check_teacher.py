"""Run a small, reproducible teacher inference check on the shipped examples.

Run from ~/Surya inside a GPU allocation:
    .venv/bin/python downstream_examples/solar_flare_forcasting/experiments/checks/check_teacher.py
"""

import json
import logging
from pathlib import Path
import random
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

EXPERIMENTS = Path(__file__).resolve().parents[1]
HERE = EXPERIMENTS.parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from infer import get_dataloader, load_model
from finetune import custom_collate_fn
from surya.utils.data import build_scalers


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Run this check inside a GPU allocation.")

    config = yaml.safe_load((HERE / "config_infer.yaml").read_text())
    for key in ("sdo_data_root_path", "valid_data_path", "flare_data_path", "scalers_path"):
        config["data"][key] = str((HERE / config["data"][key]).resolve())
    config["pretrained_path"] = str((HERE / config["pretrained_path"]).resolve())
    config["data"]["num_data_workers"] = 0
    config["data"]["scalers"] = yaml.safe_load(Path(config["data"]["scalers_path"]).read_text())
    config["dtype"] = torch.float32
    checkpoint = HERE / "assets/solar_flare_weights.pth"
    output = EXPERIMENTS / "reports" / "teacher_checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "Functionality check only; selected examples are not an accuracy benchmark.",
        "host": socket.gethostname(),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint),
        "dtype": "float32",
        "config": {**config, "dtype": "float32"},
        "results": [],
        "status": "running",
    }

    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"Results: {output / 'results.json'}", flush=True)
    try:
        print("Loading teacher using the existing strict checkpoint loader...", flush=True)
        model = load_model(config, str(checkpoint), torch.device("cuda:0"))
        model.requires_grad_(False)
        model.eval()
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        scalers = build_scalers(info=config["data"]["scalers"])
        dataset = get_dataloader(config, scalers, num_samples=3).dataset.dataset
        # Bypass __getitem__'s retry/substitution so a bad example fails visibly.
        dataset.logger = logging.getLogger("teacher_check")
        selected = []
        for label in (0, 1):
            for i, timestamp in enumerate(dataset.valid_indices):
                if int(dataset.index.loc[timestamp, "label_max"]) == label:
                    selected.append(i)
                    break
        selected += [i for i in range(len(dataset)) if i not in selected]
        selected = selected[:3]
        report["eligible_examples"] = len(dataset)
        report["selected_indices"] = selected
        print(f"Teacher loaded; {len(dataset)} eligible examples, checking indices {selected}.", flush=True)
        for index in selected:
            start = time.perf_counter()
            timestamp = str(dataset.valid_indices[index])
            print(f"Reading example {timestamp}...", flush=True)
            batch, metadata = custom_collate_fn([dataset._get_index_data(index)])
            label = int(batch["label"].item())
            # The classifier consumes only past/current observations and their time offsets.
            # The inherited loader still reads an unused future image; leave it on CPU.
            inputs = {key: batch[key].to("cuda:0") for key in ("ts", "time_delta_input")}
            del batch
            assert torch.isfinite(inputs["ts"]).all().item(), "Non-finite normalized input"
            torch.cuda.synchronize()
            preparation_seconds = time.perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                logits = model(inputs)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - start
            assert logits.numel() == 1 and torch.isfinite(logits).all().item()
            row = {
                "reference_time": timestamp,
                "input_times": [str(t) for t in metadata["timestamps_input"][0]],
                "input_shape": list(inputs["ts"].shape),
                "label": label,
                "logit": logits.item(),
                "probability": torch.sigmoid(logits).item(),
                "prediction_at_0.5": int(torch.sigmoid(logits).item() > 0.5),
                "preparation_seconds": preparation_seconds,
                "inference_seconds_unwarmed": inference_seconds,
                "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved() / 1024**3,
            }
            if index == selected[0]:
                with torch.inference_mode():
                    repeated = model(inputs)
                row["repeat_logit_absolute_difference"] = (logits - repeated).abs().max().item()
                torch.testing.assert_close(logits, repeated, rtol=1e-5, atol=1e-6)
                del repeated
            report["results"].append(row)
            save()
            print(json.dumps(row), flush=True)
            del inputs, logits
        report["status"] = "passed"
        report["logit_range"] = max(r["logit"] for r in report["results"]) - min(r["logit"] for r in report["results"])
        save()
        print("Teacher functionality check passed.", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
