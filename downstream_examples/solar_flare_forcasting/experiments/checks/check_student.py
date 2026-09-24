"""Check ResNet18 with two prepared examples and one training update.

Run from ~/Surya inside your GPU allocation:
    .venv/bin/python -u downstream_examples/solar_flare_forcasting/experiments/checks/check_student.py

Uses one negative and one positive example, FP32, and seed 42. No extra
normalization or augmentation. This is a disposable check, not a training run:
no weights are saved. Reports go to experiments/reports/<pilot>/student_checks/.
Use --student-dir to select another prepared dataset; --device cpu is optional
for environments where running the full 1024x1024 check on CPU is appropriate.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EXPERIMENTS = Path(__file__).resolve().parents[1]
TASK = EXPERIMENTS.parent
sys.path.insert(0, str(TASK.parents[1]))
from downstream_examples.solar_flare_forcasting.models import ResNet18Classifier

DEFAULT_DATA = EXPERIMENTS / "student_data/1024/pilot50_20260919T220418_102479Z"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class CheckDataset(Dataset):
    """Load the first example of each class from completed preparation metadata."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.state = json.loads((self.directory / "preparation.json").read_text())
        records = self.state["samples"]
        if self.state["status"] != "complete" or len(records) != self.state["expected_samples"]:
            raise ValueError("Finish student preparation before running this check.")
        if len({r["sample_id"] for r in records}) != len(records):
            raise ValueError("Duplicate prepared sample IDs.")
        provenance = self.state["provenance"]
        if (len(provenance["channels"]) != 13 or provenance["student_size"] != 1024
                or provenance["input_times"] != ["t-60min", "t"]):
            raise ValueError("Expected 13 channels, two chronological frames, and size 1024.")
        self.rows = []
        for label in (0, 1):
            row = next((r for r in records if int(r["label_max"]) == label), None)
            if row is None:
                raise ValueError(f"No example with label {label}.")
            if row["split"] != "train":
                raise ValueError("Use training examples for the training-update check.")
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.directory / "arrays" / f"{row['sample_id']}.npy"
        if Path(row["array_path"]).resolve() != path.resolve():
            raise ValueError(f"Array path mismatch: {row['sample_id']}")
        if sha256(path) != row["array_sha256"]:
            raise ValueError(f"Array checksum mismatch: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (13, 2, 1024, 1024) or array.dtype != np.float32:
            raise ValueError(f"Unexpected shape/dtype: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite input: {path}")
        return {"ts": torch.from_numpy(array),
                "label": torch.tensor(float(row["label_max"])), "sample_id": row["sample_id"]}


def training_step(batch, device):
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
    model = ResNet18Classifier(in_channels=13, time_steps=2, num_classes=1).to(device)
    model.train()
    inputs = batch["ts"].to(device)
    labels = batch["label"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = {name: p.detach().clone() for name, p in (
        ("first_convolution", model.resnet[0].weight), ("classifier", model.classifier.weight))}
    merged_shape = []

    def check_merge(module, args):
        merged = args[0]
        merged_shape[:] = list(merged.shape)
        # The existing model puts all 13 previous channels before all 13 current channels.
        if not (torch.equal(merged[:, :13], inputs[:, :, 0])
                and torch.equal(merged[:, 13:], inputs[:, :, 1])):
            raise ValueError("Channel/time merge changed input order.")

    hook = model.resnet[0].register_forward_pre_hook(check_merge)
    optimizer.zero_grad(set_to_none=True)
    try:
        logits = model({"ts": inputs})
    finally:
        hook.remove()
    if logits.shape != labels.shape or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite logit per example.")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite loss.")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing or non-finite gradient: {name}")
    gradient_norms = {"first_convolution": model.resnet[0].weight.grad.norm().item(),
                      "classifier": model.classifier.weight.grad.norm().item()}
    if any(value <= 0 for value in gradient_norms.values()):
        raise ValueError("Expected nonzero gradients in the backbone and classifier.")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite parameters after optimizer step.")
    changes = {name: (p.detach() - before[name]).abs().max().item() for name, p in (
        ("first_convolution", model.resnet[0].weight), ("classifier", model.classifier.weight))}
    if any(value <= 0 for value in changes.values()):
        raise ValueError("Optimizer did not update both the backbone and classifier.")
    return {"input_shape": list(inputs.shape), "merged_shape": merged_shape,
            "output_shape": list(logits.shape), "labels": labels.tolist(),
            "logits_before_update": logits.detach().tolist(),
            "probabilities_before_update": logits.detach().sigmoid().tolist(),
            "loss_before_update": loss.item(), "loss": "BCEWithLogitsLoss (true labels only)",
            "gradient_norms": gradient_norms, "maximum_weight_changes": changes,
            "all_gradients_finite": True, "all_parameters_finite_after_update": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("No CUDA device available. Run inside your GPU allocation.")
    torch.set_num_threads(2)
    directory = args.student_dir.resolve()
    output = (EXPERIMENTS / "reports" / directory.name / "student_checks"
              / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "student_dir": str(directory), "seed": 42,
              "architecture": "ResNet18Classifier", "initialization": "random; weights=None",
              "device": args.device, "torch_version": torch.__version__,
              "optimizer": "Adam", "learning_rate": 1e-4, "dropout": 0.1,
              "precision": "float32", "optimizer_steps": 1,
              "script_sha256": sha256(__file__), "model_code_sha256": sha256(TASK / "models.py")}
    print(f"Results: {output / 'results.json'}", flush=True)
    start = time.perf_counter()
    try:
        dataset = CheckDataset(directory)
        report["preparation_sha256"] = sha256(directory / "preparation.json")
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
        report["sample_ids"] = batch["sample_id"]
        print(f"Loaded {batch['sample_id']}: {list(batch['ts'].shape)}", flush=True)
        if args.device == "cuda":
            report["gpu"] = torch.cuda.get_device_name()
        report.update(training_step(batch, torch.device(args.device)))
        report.update(status="passed", elapsed_seconds=time.perf_counter() - start)
        print(f"Passed: loss={report['loss_before_update']:.6f}; finite gradients; "
              "backbone and classifier weights updated.", flush=True)
        print("One disposable update completed. No model checkpoint saved.", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
