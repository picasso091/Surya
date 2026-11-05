"""Test-set predictions of the finetuned teacher at 224.

Run from downstream_examples/solar_flare_forcasting, after teacher224_lora.py:
    CUDA_VISIBLE_DEVICES=1 conda run -n suryaenv --no-capture-output python experiments/teacher_224/teacher224_lora_test.py \
        --run_name lora_css
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from teacher224_lora import Flare224Dataset, add_lora, load_teacher_224, predict, round4, scores, skill


def main():
    cfg = yaml.safe_load(open("config_224.yaml"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=cfg["run_name"], help="results subfolder with best.pth and results.json")
    d = cfg["data"]
    out = Path(cfg["output_folder"]) / parser.parse_args().run_name

    # 1. Test data
    test = Flare224Dataset(d, d["test_flare_data_path"])
    loader = DataLoader(test, batch_size=d["batch_size"], num_workers=d["num_data_workers"])

    # 2. The finetuned model: teacher at 224 + LoRA, with the weights from best.pth
    model = add_lora(load_teacher_224(cfg), cfg)
    model.load_state_dict(torch.load(out / "best.pth", map_location="cpu"))
    model = model.cuda()

    # 3. Predict, and call a flare when the probability reaches the validation cut-off
    threshold = json.load(open(out / "results.json"))["threshold"]
    probs = predict(model, loader)
    flare = probs >= threshold

    # 4. Test scores: TSS, HSS and CSS at the validation cut-off, and ROC-AUC
    result = round4({"valid_threshold": threshold, **skill(test.labels, flare),
                     "roc_auc": scores(test.labels, probs)["roc_auc"]})
    print(json.dumps(result, indent=1))

    # 5. Save test_predictions.csv
    json.dump(result, open(out / "test_results.json", "w"), indent=1)
    test.samples.assign(prob=probs, flare=flare)[["timestamp", d["label_column"], "prob", "flare"]].to_csv(
        out / "test_predictions.csv", index=False)
    print(f"saved {out}/test_predictions.csv and test_results.json")


if __name__ == "__main__":
    main()
