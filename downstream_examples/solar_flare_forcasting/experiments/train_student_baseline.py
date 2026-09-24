"""Train a labels-only ResNet18 on the 50-example pilot.

Run from ~/Surya inside your GPU allocation:
    .venv/bin/python -u downstream_examples/solar_flare_forcasting/experiments/train_student_baseline.py

Defaults: 10 epochs, batch size 2, Adam lr=0.0001, seed 42, FP32.
Uses already-normalized 1024x1024 arrays; no augmentation or teacher targets.
All metrics describe the SAME training examples, not validation/test performance.
Epoch 0 measures the randomly initialized model before any training updates.

Outputs: experiments/runs/labels_only/<pilot>/seed42_<timestamp>/
  results.json           Settings, provenance, progress, and epoch metrics
  history.csv            Training loss and evaluation metrics by epoch
  last.pt                Latest completed epoch's model and optimizer state
  train_predictions.csv  Latest evaluation predictions on the training pilot
  student_errors_vs_teacher.csv  Latest student errors and teacher probabilities
  comparisons/epoch_NNN.csv      Error comparison at each epoch, including epoch 0

Every invocation starts a fresh run in a new folder; there is no resume option.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EXPERIMENTS = Path(__file__).resolve().parent
TASK = EXPERIMENTS.parent
sys.path.insert(0, str(TASK.parents[1]))
from downstream_examples.solar_flare_forcasting.models import ResNet18Classifier

PILOT = "pilot50_20260919T220418_102479Z"
INPUT_SHAPE = (13, 2, 1024, 1024)
COMPARISON_FIELDS = ["epoch", "sample_id", "label_max", "student_probability",
                     "teacher_probability", "student_probability_percent",
                     "teacher_probability_percent", "student_prediction_at_0.5",
                     "teacher_prediction_at_0.5"]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class StudentDataset(Dataset):
    """Use true labels from preparation metadata, checked against the manifest."""

    def __init__(self, directory, manifest):
        self.directory = Path(directory).resolve()
        self.state = json.loads((self.directory / "preparation.json").read_text())
        self.rows = self.state["samples"]
        provenance = self.state["provenance"]
        if self.state["status"] != "complete" or len(self.rows) != self.state["expected_samples"]:
            raise ValueError("A complete student preparation is required.")
        if (len(provenance["channels"]) != INPUT_SHAPE[0]
                or provenance["student_size"] != INPUT_SHAPE[-1]
                or provenance["input_times"] != ["t-60min", "t"]
                or provenance["dtype"] != "float32"):
            raise ValueError("Unexpected prepared input settings.")
        if sha256(manifest) != provenance["manifest_sha256"]:
            raise ValueError("Manifest differs from the one used for preparation.")
        with Path(manifest).open(newline="") as handle:
            originals = list(csv.DictReader(handle))
        indexed = {row["sample_id"]: row for row in originals}
        ids = [row["sample_id"] for row in self.rows]
        if len(set(ids)) != len(ids) or len(indexed) != len(originals) or set(ids) != set(indexed):
            raise ValueError("Prepared samples must match the manifest exactly, with unique IDs.")
        for row in self.rows:
            if row["split"] != "train" or row["label_max"] not in (0, 1):
                raise ValueError("Expected training examples with binary labels.")
            for key in ("split", "label_max", "reference_time_utc", "previous_time_utc", "current_time_utc"):
                if str(row[key]) != str(indexed[row["sample_id"]][key]):
                    raise ValueError(f"Manifest mismatch for {row['sample_id']}: {key}")
        if {row["label_max"] for row in self.rows} != {0, 1}:
            raise ValueError("The pilot must contain both classes.")

    def __len__(self):
        return len(self.rows)

    def path(self, row):
        path = self.directory / "arrays" / f"{row['sample_id']}.npy"
        if Path(row["array_path"]).resolve() != path.resolve():
            raise ValueError(f"Saved path mismatch: {row['sample_id']}")
        return path

    def verify_files(self):
        # Check hashes once before training, rather than rereading for every batch.
        for row in self.rows:
            path = self.path(row)
            if path.stat().st_size != row["size_bytes"] or sha256(path) != row["array_sha256"]:
                raise ValueError(f"Array size/checksum mismatch: {path}")
            self.load_array(row)

    def load_array(self, row):
        array = np.load(self.path(row), allow_pickle=False)
        if array.shape != INPUT_SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Unexpected shape, dtype, or non-finite input: {row['sample_id']}")
        return array

    def __getitem__(self, index):
        row = self.rows[index]
        return {"ts": torch.from_numpy(self.load_array(row)),
                "label": torch.tensor(float(row["label_max"])), "sample_id": row["sample_id"]}


def training_losses(logits, labels, sample_ids, teacher=None):
    label_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    if teacher is None:
        return label_loss, label_loss, None
    # Binary soft-target cross-entropy at temperature 1. Targets are constants
    # looked up by ID, so shuffled batches cannot misalign teacher predictions.
    targets = logits.new_tensor([teacher[sid]["teacher_probability"] for sid in sample_ids])
    teacher_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return 0.5 * label_loss + 0.5 * teacher_loss, label_loss, teacher_loss


def run_epoch(model, loader, device, optimizer=None, teacher=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count, correct = 0., 0, 0
    label_loss_sum, teacher_loss_sum = 0., 0.
    predictions = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = batch["ts"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model({"ts": inputs})
            if logits.shape != labels.shape or not torch.isfinite(logits).all():
                raise ValueError("Expected one finite logit per example.")
            loss, label_loss, teacher_loss = training_losses(logits, labels, batch["sample_id"], teacher)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss.")
            if training:
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                if any(g is None for g in gradients) or not torch.stack([
                        torch.isfinite(g).all() for g in gradients]).all():
                    raise ValueError("Missing or non-finite gradients.")
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            label_loss_sum += label_loss.item() * len(labels)
            if teacher_loss is not None:
                teacher_loss_sum += teacher_loss.item() * len(labels)
            count += len(labels)
            if not training:
                probabilities = logits.sigmoid()
                predicted = probabilities >= 0.5
                correct += (predicted == labels.bool()).sum().item()
                for sid, label, logit, probability, prediction in zip(
                        batch["sample_id"], labels.tolist(), logits.tolist(),
                        probabilities.tolist(), predicted.tolist()):
                    predictions.append({"sample_id": sid, "label_max": int(label),
                                        "student_logit": logit, "student_probability": probability,
                                        "prediction_at_0.5": int(prediction)})
    if count != len(loader.dataset):
        raise ValueError("An epoch must visit every pilot example exactly once.")
    return {"loss": loss_sum / count, "label_loss": label_loss_sum / count,
            "teacher_loss": teacher_loss_sum / count if teacher is not None else None,
            "accuracy": correct / count if not training else None}, predictions


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows, fieldnames=None):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_teacher_comparison(dataset, cache_path):
    """Read matched cached probabilities; baseline uses these only for reporting."""
    if sha256(cache_path) != dataset.state["provenance"]["teacher_cache_sha256"]:
        raise ValueError("Teacher cache differs from the one used for student preparation.")
    cache = json.loads(cache_path.read_text())
    records = cache["predictions"]
    teacher = {row["sample_id"]: row for row in records}
    if (cache["status"] != "complete" or len(teacher) != len(records)
            or set(teacher) != {row["sample_id"] for row in dataset.rows}):
        raise ValueError("Expected a complete teacher cache with matching sample IDs.")
    for row in dataset.rows:
        target = teacher[row["sample_id"]]
        if any(str(target[key]) != str(row[key]) for key in ("label_max", "split", "reference_time_utc")):
            raise ValueError(f"Teacher metadata mismatch: {row['sample_id']}")
        probability = float(target["teacher_probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid teacher probability: {row['sample_id']}")
    return teacher


def save_error_comparison(output, epoch, predictions, teacher):
    rows = []
    for prediction in predictions:
        label = prediction["label_max"]
        student_p = prediction["student_probability"]
        if int(student_p >= 0.5) == label:
            continue
        teacher_p = float(teacher[prediction["sample_id"]]["teacher_probability"])
        rows.append({"epoch": epoch, "sample_id": prediction["sample_id"], "label_max": label,
                     "student_probability": student_p, "teacher_probability": teacher_p,
                     "student_probability_percent": round(100 * student_p, 1),
                     "teacher_probability_percent": round(100 * teacher_p, 1),
                     "student_prediction_at_0.5": int(student_p >= 0.5),
                     "teacher_prediction_at_0.5": int(teacher_p >= 0.5)})
    (output / "comparisons").mkdir(exist_ok=True)
    # A zero-error epoch writes a header-only CSV, replacing any previous errors.
    save_csv(output / "comparisons" / f"epoch_{epoch:03d}.csv", rows, COMPARISON_FIELDS)
    save_csv(output / "student_errors_vs_teacher.csv", rows, COMPARISON_FIELDS)


def save_checkpoint(path, model, optimizer, epoch, settings, generator):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "settings": settings,
                "torch_rng_state": torch.get_rng_state(),
                "shuffle_rng_state": generator.get_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary)
    temporary.replace(path)


def main(*, distillation=False, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-dir", type=Path, default=EXPERIMENTS / "student_data/1024" / PILOT)
    parser.add_argument("--manifest", type=Path, default=EXPERIMENTS / "manifests" / PILOT / "manifest.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("No CUDA device available. Run inside your GPU allocation.")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    mode = "distillation" if distillation else "labels_only"
    output = (EXPERIMENTS / "runs" / mode / args.manifest.parent.name
              / f"seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}")
    output.mkdir(parents=True, exist_ok=False)
    settings = {**vars(args), "student_dir": str(args.student_dir.resolve()),
                "manifest": str(args.manifest.resolve()), "architecture": "ResNet18Classifier",
                "initialization": "random; weights=None", "optimizer": "Adam", "dropout": 0.1,
                "precision": "float32", "loss": "0.5 * label BCE + 0.5 * teacher BCE" if distillation else "BCEWithLogitsLoss",
                "teacher_targets_used": distillation, "distillation_weight": 0.5 if distillation else 0.,
                "temperature": 1.,
                "augmentation": False, "threshold": 0.5, "torch_version": str(torch.__version__),
                "numpy_version": np.__version__, "script_sha256": sha256(__file__),
                "model_code_sha256": sha256(TASK / "models.py"),
                "cudnn_deterministic": True, "tf32": False}
    if distillation:
        settings["entrypoint_sha256"] = sha256(Path(__file__).with_name("train_student_distillation.py"))
    report = {"status": "running", "settings": settings, "history": [],
              "metric_scope": "Training pilot only; no validation or test set used."}
    print(f"Run directory: {output}", flush=True)
    start = time.perf_counter()
    try:
        save_json(output / "results.json", report)
        dataset = StudentDataset(args.student_dir, args.manifest)
        teacher_path = EXPERIMENTS / "teacher_cache" / args.manifest.parent.name / "cache.json"
        teacher_comparison = load_teacher_comparison(dataset, teacher_path)
        settings.update(teacher_comparison_cache=str(teacher_path.resolve()),
                        teacher_comparison_cache_sha256=sha256(teacher_path))
        print(f"Verifying {len(dataset)} saved arrays and their labels...", flush=True)
        dataset.verify_files()
        settings.update(preparation_sha256=sha256(args.student_dir / "preparation.json"),
                        manifest_sha256=sha256(args.manifest),
                        channels=dataset.state["provenance"]["channels"], input_shape=list(INPUT_SHAPE))
        report.update(examples=len(dataset), positives=sum(r["label_max"] for r in dataset.rows))
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  generator=generator, num_workers=0)
        eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 generator=torch.Generator().manual_seed(args.seed))
        model = ResNet18Classifier(in_channels=13, time_steps=2, num_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        if device.type == "cuda":
            report["gpu"] = torch.cuda.get_device_name(device)
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(args.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, device, optimizer,
                                      teacher=teacher_comparison if distillation else None)[0] if epoch else None
            # Always evaluate against TRUE labels for comparison with the baseline.
            evaluation, predictions = run_epoch(model, eval_loader, device)
            entry = {"epoch": epoch, "train_loss": train_metrics["loss"] if epoch else None,
                     "train_eval_loss": evaluation["loss"], "train_eval_accuracy_at_0.5": evaluation["accuracy"],
                     "epoch_seconds": time.perf_counter() - epoch_start}
            if distillation:
                entry.update(train_label_loss=train_metrics["label_loss"] if epoch else None,
                             train_teacher_loss=train_metrics["teacher_loss"] if epoch else None)
            if epoch:
                save_checkpoint(output / "last.pt", model, optimizer, epoch, settings, generator)
            save_csv(output / "train_predictions.csv", predictions)
            save_error_comparison(output, epoch, predictions, teacher_comparison)
            report["history"].append(entry)
            report["completed_epochs"] = epoch
            save_csv(output / "history.csv", report["history"])
            save_json(output / "results.json", report)
            loss_text = f"{entry['train_loss']:.5f}" if epoch else "--"
            component_text = (f"label loss={train_metrics['label_loss']:.5f}, "
                              f"teacher loss={train_metrics['teacher_loss']:.5f}, ") if distillation and epoch else ""
            print(f"Epoch {epoch:2d}/{args.epochs}: train loss={loss_text}, "
                  f"{component_text}"
                  f"train-set eval loss={evaluation['loss']:.5f}, "
                  f"accuracy={evaluation['accuracy']:.1%}, {entry['epoch_seconds']:.1f}s", flush=True)
        report.update(status="complete", elapsed_seconds=time.perf_counter() - start,
                      peak_allocated_gpu_gib=torch.cuda.max_memory_allocated(device) / 1024**3
                      if device.type == "cuda" else None)
        save_json(output / "results.json", report)
        print(f"Complete. Checkpoint: {output / 'last.pt'}", flush=True)
        print("These are training-pilot metrics, not generalization results.", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            save_json(output / "results.json", report)
        except OSError as save_error:
            print(f"Could not save failure report: {save_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
