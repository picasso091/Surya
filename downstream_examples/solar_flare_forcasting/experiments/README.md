# Solar-flare distillation experiments

We have prepared a 50-example training pilot and checked the teacher and data
loader. Predictions for all 50 examples have been cached, and all 50 student
inputs have been prepared and verified at 1024x1024. The ResNet18 training-update
check and 10-epoch labels-only baseline passed on the GH200. The next step is
the matched 10-epoch distillation pilot, using the cached teacher probabilities.

## Folder guide

```text
experiments/
  README.md                   Start here: workflow and commands
  build_pilot_manifest.py     Select examples and check remote file sizes
  download_pilot_data.py      Download/resume and validate selected images
  cache_teacher_predictions.py Save teacher logits and probabilities for the pilot
  prepare_student_data.py     Downsample, normalize, and verify 1024x1024 inputs
  train_student_baseline.py   Train ResNet18 on pilot labels for 10 epochs
  train_student_distillation.py Same training loop with equal label/teacher loss
  flare_dataset.py            Shared two-input flare dataset loader
  checks/
    check_teacher.py          Verify teacher loading and example predictions
    check_flare_dataset.py    Verify one normalized example and its label
    check_student.py          Check two prepared inputs and one ResNet18 update
  tests/
    test_flare_dataset.py     Fast tests using small synthetic images
    test_teacher_cache.py     Cache integrity and interruption/resume tests on CPU
    test_student_preparation.py Block means, saved arrays, and corruption checks
    test_student_baseline.py  Training, checkpoint reload, and data integrity tests
    test_student_distillation.py Loss gradients, target alignment, and matched runs
  manifests/
    pilot50_<timestamp>/      manifest.csv, files.csv, summary.json, missing_files.csv
  data/
    core_sdo/<year>/<month>/  Downloaded images; shared across runs
  teacher_cache/
    <manifest-name>/         predictions.csv and cache.json (created when run)
  student_data/
    1024/<manifest-name>/    arrays/*.npy, samples.csv, preparation.json
  runs/
    labels_only/<pilot>/<seed_timestamp>/  Metrics, predictions, and last.pt
    distillation/<pilot>/<seed_timestamp>/ Same outputs plus both loss components
  reports/
    teacher_checks/<run>/     Teacher diagnostic results
    <manifest-name>/
      download_checks/<run>/ Download and validation results
      dataset_checks/<run>/  Loader verification results
      student_checks/<run>/  Student input, gradient, and weight-update results
```

The manifest describes which examples to use. The data folder holds the actual
images. Reports describe what a particular run checked. Keep the timestamped
manifest directory names stable: saved reports and commands refer to them.

## Current pilot

- Manifest: `manifests/pilot50_20260919T220418_102479Z/manifest.csv`.
- 50 examples: 25 positive and 25 negative; 10 examples per year, 2011–2015.
- Each example uses images at `t-60 minutes` and `t`, with a 24-hour flare label.
- Forecast times are at least 48 hours apart. This is a balanced debugging set,
  not a representative evaluation set.
- 100 images, 58,522,349,952 bytes (58.52 GB / 54.50 GiB), downloaded and validated.
- Latest retained download report:
  `reports/pilot50_20260919T220418_102479Z/download_checks/20260920T032412_632166Z/results.json`.
- Latest retained dataset report:
  `reports/pilot50_20260919T220418_102479Z/dataset_checks/20260920T035646_726100Z/results.json`.
- Latest retained teacher report:
  `reports/teacher_checks/20260919T210500Z/results.json`.

## Commands

Run from the repository root (`~/Surya`). These shell variables just shorten the
commands below; each script also contains a full command at the top.

```bash
experiment_dir=downstream_examples/solar_flare_forcasting/experiments
pilot_manifest_dir="$experiment_dir/manifests/pilot50_20260919T220418_102479Z"

# Check the loader on one example (CPU):
.venv/bin/python "$experiment_dir/checks/check_flare_dataset.py" --manifest-dir "$pilot_manifest_dir"

# Run fast synthetic-data tests (CPU):
.venv/bin/python "$experiment_dir/tests/test_flare_dataset.py"
.venv/bin/python "$experiment_dir/tests/test_teacher_cache.py"

# Download/validate all 100 files; reuse existing files and resume partials (CPU):
.venv/bin/python "$experiment_dir/download_pilot_data.py" --manifest-dir "$pilot_manifest_dir" --all

# Verify the teacher on the shipped inference examples (requires a GPU allocation):
.venv/bin/python "$experiment_dir/checks/check_teacher.py"

# Save teacher predictions for all 50 pilot examples (requires a GPU allocation):
.venv/bin/python -u "$experiment_dir/cache_teacher_predictions.py" --manifest-dir "$pilot_manifest_dir"

# Prepare and check one student example at 1024x1024 (CPU):
.venv/bin/python -u "$experiment_dir/prepare_student_data.py" --manifest-dir "$pilot_manifest_dir"
# After that passes, prepare the remaining examples (CPU; approximately 5.08 GiB total):
.venv/bin/python -u "$experiment_dir/prepare_student_data.py" --manifest-dir "$pilot_manifest_dir" --all

# Check ResNet18 on one negative and one positive example (GPU allocation):
.venv/bin/python -u "$experiment_dir/checks/check_student.py"

# Train the labels-only pilot baseline for 10 epochs (GPU allocation):
.venv/bin/python -u "$experiment_dir/train_student_baseline.py"

# Train a fresh student with equal label/teacher loss (GPU allocation):
.venv/bin/python -u "$experiment_dir/train_student_distillation.py"

# Create a NEW manifest with the same selection rules (network metadata checks only):
.venv/bin/python "$experiment_dir/build_pilot_manifest.py"
```

Running the downloader without `--all` checks only the first example (two images).
Keep `.part` files until their downloads finish; they contain resumable progress.
All scripts derive their data/config locations from their own file location.

Teacher caching saves after each sample; rerun the same command to resume. It
uses the two-input manifest dataset, batch size one, FP32, evaluation mode, and
frozen weights. `cache.json` records progress and hashes identifying the inputs,
weights, configuration, normalization, and source code. `predictions.csv` is the
readable sample-by-sample export. Changed provenance requires a new output
directory using `--output-dir experiments/teacher_cache/<new-name>` (include the
full repository-relative experiments path when running from the repository root).
The script refuses concurrent writers to the same cache. Budget approximately
15–20 minutes on the GH200, excluding scheduler queue time; actual I/O varies.
The cached probabilities are training targets, not test-set performance results.

Student preparation averages each raw 4x4 block before applying the original
channel-specific normalization. Each `.npy` stores one float32 example with shape
`[13, 2, 1024, 1024]`; do not normalize these arrays again during training.
`samples.csv` connects array paths, timestamps, true labels, and teacher logits by
sample ID. `preparation.json` records preprocessing settings, checksums, and
verification results. The script verifies completed arrays before reusing them,
and refuses mismatched manifests, teacher caches, or preprocessing settings.

The student check reuses `ResNet18Classifier` from the task's `models.py`, with
random initialization and 26 input channels (13 channels at each of two times).
It checks saved-array hashes, runs a batch of two through the model, computes
binary cross-entropy from raw logits and true labels, and takes one Adam step.
It verifies finite gradients and changes to both the first convolution and the
classifier. Results are diagnostic only: the model is discarded, and teacher
logits are not used yet. The full-resolution GPU check must pass before moving
on to baseline training.

The baseline uses all 50 prepared training examples, batch size 2, Adam with
learning rate 0.0001, seed 42, and FP32. No teacher targets, augmentation, or
additional normalization are applied. It evaluates the randomly initialized
model at epoch 0, then trains and evaluates once per epoch. `train_loss` is the
average loss during updates; `train_eval_loss` and `train_eval_accuracy_at_0.5`
use the same training examples in evaluation mode (dropout off, BatchNorm using
running statistics). These measure fit to this pilot, not generalization.

Each invocation creates a fresh timestamped directory under `runs/labels_only/`.
`results.json` contains settings, source/data hashes, progress, and epoch metrics;
`history.csv` contains the learning history. `last.pt` is replaced atomically
after every completed epoch and contains model and optimizer states. Latest
training predictions go into `train_predictions.csv` after each evaluation.
`student_errors_vs_teacher.csv` is updated at the same time with the student's
misclassified examples and both models' probabilities (including percentages).
`comparisons/epoch_000.csv`, `epoch_001.csv`, etc. preserve each epoch's comparison.
A zero-error epoch produces a header-only comparison. Teacher probabilities come
from the preparation-matched cache and are used only for reporting, not the loss.
There is no resume option
or selection of a best checkpoint based on these training metrics. Epoch times
include training and evaluation, but exclude checkpoint saving. Allow extra time
for startup verification and evaluation beyond the earlier training-only estimate.

## First distillation run

The first distillation run uses the baseline training loop with the same default
initialization seed, shuffling, optimizer, batch size, and 10-epoch budget. It
starts from random weights, not the baseline checkpoint. At temperature 1,
the loss is `0.5 * BCEWithLogits(student_logit, label)` plus
`0.5 * BCEWithLogits(student_logit, cached_teacher_probability)`. The teacher is
not loaded or updated. Both positive and negative class probabilities are
accounted for by binary cross-entropy. The teacher term is soft-target BCE,
not a reported KL divergence; it need not reach zero even at perfect agreement.

Distillation outputs use a separate `runs/distillation/` directory, including
automatic per-epoch error comparisons. `train_loss` is the combined loss;
`train_label_loss` and `train_teacher_loss` record its two unweighted components.
Compare `train_eval_loss` and `train_eval_accuracy_at_0.5` between runs: these
always use true labels in evaluation mode. Combined distillation loss and
baseline training loss are different objectives. All results still concern the
same 50 training examples and do not measure generalization.

## Keeping this readable

- Keep workflow scripts and shared code at the top level while there are only a
  few. Put diagnostics in `checks/` and automated tests in `tests/`.
- Store teacher predictions in `teacher_cache/<manifest-name>/`, and student
  training artifacts in `runs/<experiment-name>/<seed>/` when needed.
- Retain reports for meaningful experiments, including failures that explain a
  problem. Repeated setup checks can be removed once their results are covered
  by a later successful check.
- Version the Python files and this README. `.gitignore` excludes downloaded
  data, generated manifests, reports, and bytecode. Back up the selected manifest
  and results separately; they are needed to reproduce the study.

## Cleanup performed

Kept all 100 images, the four manifest files, all six Python tools/tests, and one
successful report for each completed setup task. Removed six redundant setup
reports: four partial/repeated download checks covered by the complete 100-file
check, one repeated dataset check, and one repeated teacher check with identical
prediction scores. Removed generated `__pycache__` files. Existing training and
inference implementations were not reorganized.
