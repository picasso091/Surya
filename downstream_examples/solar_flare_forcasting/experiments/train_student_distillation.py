"""Train a fresh ResNet18 using true labels and cached teacher probabilities.

Run from ~/Surya inside your GPU allocation:
    .venv/bin/python -u downstream_examples/solar_flare_forcasting/experiments/train_student_distillation.py

First pilot: loss = 0.5 * BCE(student_logit, true_label)
                 + 0.5 * BCE(student_logit, teacher_probability).
Both BCE terms use binary_cross_entropy_with_logits. Temperature is fixed at 1.
Soft-target BCE differs from Bernoulli KL by teacher entropy (a constant with
respect to the student); the logged teacher loss need not reach zero.

Defaults match the labels-only baseline: 10 epochs, batch size 2, Adam lr=0.0001,
seed 42, dropout 0.1, FP32, identical inputs and shuffle. No checkpoint is loaded.
The shared training loop preserves initialization and data-order behavior.
Teacher inference is not repeated; frozen probabilities come from cache.json.

Outputs: experiments/runs/distillation/<pilot>/seed42_<timestamp>/
  history.csv and results.json: total, true-label, and teacher training losses;
      evaluation loss/accuracy are always computed against the true labels.
  last.pt: latest completed epoch's model and optimizer.
  train_predictions.csv: latest predictions on all 50 training examples.
  student_errors_vs_teacher.csv: latest error comparison.
  comparisons/epoch_NNN.csv: error comparisons for each epoch, including epoch 0.

These are training-pilot metrics, not validation or test results. Use true-label
evaluation loss to compare runs; baseline and distillation total losses differ.
"""

from train_student_baseline import main


if __name__ == "__main__":
    main(distillation=True, description=__doc__)
