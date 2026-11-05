"""Print the 4k model's parameter counts and the tensor shape after each stage.
    python downstream_examples/solar_flare_forcasting/experiments/checks/show_4k_arch.py
"""

import sys
from pathlib import Path

import torch
import yaml

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parents[1]))

from models import HelioSpectformer1D  # noqa: E402
from finetune import apply_peft_lora  # noqa: E402

config = yaml.safe_load((TASK / "config_infer.yaml").read_text())
m = config["model"]

# 1. Build the teacher exactly as configured (random weights)
model = HelioSpectformer1D(
    img_size=m["img_size"], patch_size=m["patch_size"], in_chans=m["in_channels"],
    embed_dim=m["embed_dim"], time_embedding=m["time_embedding"], depth=m["depth"],
    n_spectral_blocks=m["spectral_blocks"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
    drop_rate=m["drop_rate"], window_size=m["window_size"], dp_rank=m["dp_rank"],
    learned_flow=m["learned_flow"], checkpoint_layers=[], rpe=m["rpe"], ensemble=m["ensemble"],
    finetune=m["finetune"], nglo=m["nglo"], dtype=torch.float32, dropout=m["dropout"],
    num_outputs=1, num_penultimate_transformer_layers=0, num_penultimate_heads=0, config=config,
).eval()

# 2. Record each stage's input/output shape during the forward pass
stages = [("embedding", model.embedding)]
stages += [(f"spectral_block[{i}]", b) for i, b in enumerate(model.backbone.blocks_spectral_gating)]
stages += [(f"attention_block[{i}]", b) for i, b in enumerate(model.backbone.blocks_attention)]
stages += [("linear (penultimate)", model.linear), ("unembed (head)", model.unembed)]

shapes = []
for name, stage in stages:
    stage.register_forward_hook(
        lambda mod, inp, out, name=name: shapes.append((name, tuple(inp[0].shape), tuple(out.shape))))

# 3. Add LoRA adapters, as in the released teacher, and count parameters
model = apply_peft_lora(model, config)
total = sum(p.numel() for p in model.parameters())
lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
print("\n=== Parameters ===")
print(f"total {total / 1e6:.1f} M   LoRA adapters {lora / 1e6:.2f} M   base {(total - lora) / 1e6:.1f} M")

# 4. One dummy forward pass. Input shape: (batch, channels, time steps, H, W)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
size, time_steps = m["img_size"], config["data"]["n_input_timestamps"]
batch = {"ts": torch.zeros(1, m["in_channels"], time_steps, size, size, device=device), #called "dummy" because the input is all zeros
         "time_delta_input": torch.zeros(1, time_steps, device=device)}
with torch.inference_mode():
    logit = model(batch)

grid = size // m["patch_size"]
print(f"\n=== Shapes (device {device}) ===")
print(f"patch grid {grid}x{grid} = {grid * grid:,} tokens of dim {m['embed_dim']}; "
      f"CLS token is prepended before the first attention block")
print(f"{'stage':24s}{'input':>32s}{'output':>24s}")
print(f"{'batch[ts]':24s}{str(tuple(batch['ts'].shape)):>32s}")
for name, inp, out in shapes:
    print(f"{name:24s}{str(inp):>32s}{str(out):>24s}")
print(f"{'final logit':24s}{'':>32s}{str(tuple(logit.shape)):>24s}")
