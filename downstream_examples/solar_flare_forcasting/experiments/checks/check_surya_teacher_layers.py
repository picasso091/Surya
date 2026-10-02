"""Show the layers of the Surya base model and of the solar-flare teacher side by side.
Run : conda run -n suryaenv python downstream_examples/solar_flare_forcasting/experiments/checks/check_surya_teacher_layers.py
"""
# even though the finetune says target modules = ["q_proj", "v_proj", "k_proj", "out_proj"], the teacher's state_dict has "qkv" and "proj" keys, so the LoRA layers are not actually used in attention layers. is this intentional while finetuning the teacher model?
import argparse
import re
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[2]


def load(path):
    """Return {key: (full name, tensor)}.

    The teacher's names have PEFT's base_model.model. prefix, and .base_layer. on
    LoRA-wrapped layers. The key drops both so the same layer of both files shares a row.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    return {
        name.removeprefix("module.").removeprefix("base_model.model.")
            .replace(".base_layer.", ".").replace(".default.", "."): (name, tensor)
        for name, tensor in state.items()
    }


def merge_blocks(name):
    """blocks_attention.3.norm1.weight -> blocks_attention.*.norm1.weight"""
    return re.sub(r"\.\d+\.", ".*.", name)


def section(name):
    if name.startswith("embedding"):
        return "1. EMBEDDING"
    if "blocks_spectral_gating" in name:
        return "2. SPECTRAL BLOCKS"
    if "blocks_attention" in name:
        return "3. ATTENTION BLOCKS"
    return "4. OUTPUT: head (teacher) / decoder (Surya)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--surya", default=TASK / "assets/surya.366m.v1.pt")
    parser.add_argument("--teacher", default=TASK / "assets/solar_flare_weights.pth")
    args = parser.parse_args()
    files = {"surya": load(args.surya), "teacher": load(args.teacher)}

    # One row per layer; a layer repeated in every block is one row
    rows = {}
    for source, state in files.items():
        for key, (name, tensor) in state.items():
            row = rows.setdefault(merge_blocks(key), {"shape": tuple(tensor.shape), "blocks": set(),
                                                      "surya": "-", "teacher": "-", "order": len(rows)})
            row[source] = merge_blocks(name)
            row["blocks"].add(key)

    width = max(len(row["surya"]) for row in rows.values()) + 2
    current = None
    for group, row in sorted(rows.items(), key=lambda item: (section(item[0]), item[1]["order"])):
        if section(group) != current:
            current = section(group)
            print(f"\n{current}\n{'blocks':>6s}  {'shape':22s}{'Surya':{width}s}Teacher")
        print(f"{len(row['blocks']):>6d}  {str(row['shape']):22s}{row['surya']:{width}s}{row['teacher']}")

    print("\nTOTALS")
    for source, state in files.items():
        total = sum(tensor.numel() for _, tensor in state.values())
        lora = sum(tensor.numel() for name, tensor in state.values() if ".lora_" in name)
        print(f"  {source:8s} {total / 1e6:7.1f} M numbers ({lora / 1e6:.2f} M LoRA)")
    with_lora = sorted({merge_blocks(name.split(".lora_")[0])
                        for name, _ in files["teacher"].values() if ".lora_" in name})
    print(f"  teacher layers with LoRA: {with_lora}")


if __name__ == "__main__":
    main()
