#!/usr/bin/env python3
"""Transfer weights from bimanual lift model to pick_place model.

The lift model has 63-dim observation, pick_place has 64-dim.
The first layer weight matrix is expanded: first 63 cols copied, last col random init.
All other layers are identical and copied directly.
"""
import sys
import torch
from pathlib import Path

LIFT_CKPT = Path("/home/kk/openarm_isaac_lab-1/models/stage_1_models/model_best_goal_13.47_iter500.pt")
OUT_CKPT = Path("/home/kk/openarm_isaac_lab-1/models/stage_1_models/model_best_goal_13.47_iter500_transfer.pt")

def _expand_63_to_64(t):
    """Expand tensor from shape (*, 63) to (*, 64), padding last dim with zeros."""
    if t.ndim >= 2 and t.shape[-1] == 63:
        new_shape = list(t.shape)
        new_shape[-1] = 64
        new_t = torch.zeros(new_shape, dtype=t.dtype, device=t.device)
        new_t[..., :63] = t
        return new_t
    return t


def main():
    src = torch.load(LIFT_CKPT, map_location="cpu")

    # Expand model weights
    src_state = src["model_state_dict"]
    dst_state = {}
    for key, src_tensor in src_state.items():
        if key in ("actor.0.weight", "critic.0.weight") and src_tensor.shape[1] == 63:
            new_t = _expand_63_to_64(src_tensor)
            dst_state[key] = new_t
            print(f"[EXPAND model] {key}: {src_tensor.shape} → {new_t.shape}")
        else:
            dst_state[key] = src_tensor

    # Expand optimizer state
    if "optimizer_state_dict" in src:
        opt_state = src["optimizer_state_dict"]["state"]
        for pid, param_state in opt_state.items():
            for k in list(param_state.keys()):
                t = param_state[k]
                new_t = _expand_63_to_64(t)
                if new_t is not t:
                    param_state[k] = new_t
                    print(f"[EXPAND optim] param {pid}.{k}: {t.shape} → {new_t.shape}")

    new_src = {k: v for k, v in src.items() if k != "model_state_dict"}
    new_src["model_state_dict"] = dst_state

    torch.save(new_src, OUT_CKPT)
    print(f"\nSaved: {OUT_CKPT}")
    print("Use this checkpoint with --load_run and --load_checkpoint")

if __name__ == "__main__":
    main()
