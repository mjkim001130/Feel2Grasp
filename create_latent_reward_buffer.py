#!/usr/bin/env python3
"""
Create replay buffer with latent-based reward.

Based on Replay_buffer.ipynb + reward_latent.py algorithm.
Uses existing replay_buffer_iql_72d.npz and adds latent reward from left/right encoders.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from typing import Dict

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ======================== Encoder ========================
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def load_encoder(path: str, device: torch.device) -> tuple:
    """Load encoder and return (model, resize_hw, latent_dim)"""
    ckpt = torch.load(path, map_location="cpu")
    latent_dim = ckpt["latent_dim"]
    resize_hw = tuple(ckpt.get("resize_hw", (128, 128)))

    model = ConvEncoder(latent_dim)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, resize_hw, latent_dim


# ======================== Utilities ========================
def latent_reward(
    z: np.ndarray,
    ep: np.ndarray,
    base_z_by_ep: Dict[int, np.ndarray],
    circle_z_by_ep: Dict[int, np.ndarray],
    global_circle_z: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute latent-based reward.

    z: (N, D) latent vectors
    ep: (N,) episode indices
    base_z_by_ep: dict mapping episode -> first frame latent (base/starting position)
    circle_z_by_ep: dict mapping episode -> mean latent of success frames
    global_circle_z: fallback mean latent for episodes without success frames

    Returns: (N,) reward in [-1, 1]
    """
    N, D = z.shape
    r = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        e = int(ep[i])
        z_base = base_z_by_ep[e]
        z_circle = circle_z_by_ep.get(e, global_circle_z)

        d_base = np.linalg.norm(z[i] - z_base)
        d_circle = np.linalg.norm(z[i] - z_circle)

        # Reward: closer to circle = higher, closer to base = lower
        r[i] = (d_base - d_circle) / (d_base + d_circle + eps)

    return r


@torch.no_grad()
def encode_images(
    ds: LeRobotDataset,
    ds_indices: np.ndarray,
    image_key: str,
    encoder: nn.Module,
    resize_hw: tuple,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode images to latent vectors."""
    N = len(ds_indices)
    latent_dim = encoder.fc.out_features
    Z = np.empty((N, latent_dim), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc=f"Encode {image_key}"):
        end = min(N, start + batch_size)
        idx_batch = ds_indices[start:end]

        imgs = []
        for i in idx_batch:
            sample = ds[int(i)]
            img = sample[image_key]

            # Convert to tensor if numpy
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[-1] == 3:  # HWC
                    img = torch.from_numpy(img).permute(2, 0, 1)
                else:
                    img = torch.from_numpy(img)

            # Normalize to [0, 1]
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            else:
                img = img.float()
                if img.max() > 1.5:
                    img = img / 255.0

            imgs.append(img)

        x = torch.stack(imgs, dim=0).to(device)
        x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
        z = encoder(x)
        Z[start:end] = z.cpu().numpy()

    return Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="mjkim00/Feel2Grasp")
    parser.add_argument("--parquet", default="./train.parquet")
    parser.add_argument("--front_encoder", default="ae_out/encoder.pt")
    parser.add_argument("--left_encoder", default="ae_side_out/left_encoder.pt")
    parser.add_argument("--right_encoder", default="ae_side_out/right_encoder.pt")
    parser.add_argument("--out_npz", default="replay_buffer_latent_reward.npz")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--reward_threshold", type=float, default=1.0,
                        help="circle_reward >= this value is considered success")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoders
    print("Loading encoders...")
    front_enc, front_hw, front_dim = load_encoder(args.front_encoder, device)
    left_enc, left_hw, left_dim = load_encoder(args.left_encoder, device)
    right_enc, right_hw, right_dim = load_encoder(args.right_encoder, device)
    print(f"  Front: {front_dim}d, Left: {left_dim}d, Right: {right_dim}d")

    # Load parquet
    print("Loading parquet...")
    df = pd.read_parquet(args.parquet)

    need_cols = ["action", "observation.state", "episode_index", "frame_index", "index",
                 "left_image_circle", "right_image_circle", "circle_reward"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    N = len(df)
    print(f"N = {N}")

    ep = df["episode_index"].to_numpy()
    fr = df["frame_index"].to_numpy()
    ds_indices = df["index"].to_numpy().astype(np.int64)

    obs_state = np.stack(df["observation.state"].to_list()).astype(np.float32)
    actions = np.stack(df["action"].to_list()).astype(np.float32)
    lr01 = df[["left_image_circle", "right_image_circle"]].to_numpy().astype(np.float32)
    circle_reward_raw = df["circle_reward"].to_numpy().astype(np.float32)

    # Success mask
    success_mask = circle_reward_raw >= args.reward_threshold - 1e-6
    print(f"Success frames: {success_mask.sum()} / {N} ({success_mask.mean()*100:.2f}%)")

    # Load dataset
    print("Loading dataset...")
    ds = LeRobotDataset(args.repo_id, revision="main", video_backend="pyav")
    print(f"Dataset length: {len(ds)}")

    # Encode images
    print("Encoding front images...")
    Z_front = encode_images(ds, ds_indices, "observation.images.front",
                            front_enc, front_hw, device, args.batch_size)

    print("Encoding left images...")
    Z_left = encode_images(ds, ds_indices, "observation.images.left",
                           left_enc, left_hw, device, args.batch_size)

    print("Encoding right images...")
    Z_right = encode_images(ds, ds_indices, "observation.images.right",
                            right_enc, right_hw, device, args.batch_size)

    # Build base/circle latents per episode
    print("Computing latent rewards...")
    base_left_by_ep: Dict[int, np.ndarray] = {}
    base_right_by_ep: Dict[int, np.ndarray] = {}
    circle_left_by_ep: Dict[int, np.ndarray] = {}
    circle_right_by_ep: Dict[int, np.ndarray] = {}

    # Global fallback (mean of all success frames)
    if success_mask.any():
        global_circle_left = Z_left[success_mask].mean(axis=0)
        global_circle_right = Z_right[success_mask].mean(axis=0)
    else:
        global_circle_left = Z_left.mean(axis=0)
        global_circle_right = Z_right.mean(axis=0)

    # Per-episode aggregation
    start = 0
    while start < N:
        e = int(ep[start])
        end = start
        while end < N and int(ep[end]) == e:
            end += 1

        # First frame = base
        base_left_by_ep[e] = Z_left[start].copy()
        base_right_by_ep[e] = Z_right[start].copy()

        # Success frames in this episode
        m = success_mask[start:end]
        if m.any():
            circle_left_by_ep[e] = Z_left[start:end][m].mean(axis=0)
            circle_right_by_ep[e] = Z_right[start:end][m].mean(axis=0)

        start = end

    # Compute latent reward
    r_left = latent_reward(Z_left, ep, base_left_by_ep, circle_left_by_ep, global_circle_left)
    r_right = latent_reward(Z_right, ep, base_right_by_ep, circle_right_by_ep, global_circle_right)

    # Average to get [-1, 1] range
    rewards = (r_left + r_right) / 2.0

    print(f"Reward stats: min={rewards.min():.4f}, mean={rewards.mean():.4f}, max={rewards.max():.4f}")

    # Build states: [front_z(64), left_z(16), right_z(16), obs_state(6)] = 102d
    states = np.concatenate([Z_front, Z_left, Z_right, obs_state], axis=1).astype(np.float32)
    print(f"State dim: {states.shape[1]}")

    # Terminals
    terminals = np.zeros((N,), dtype=np.float32)
    cont = (ep[1:] == ep[:-1]) & (fr[1:] == fr[:-1] + 1)
    terminals[:-1] = (~cont).astype(np.float32)
    terminals[-1] = 1.0

    # Next states
    next_states = np.empty_like(states)
    next_states[:-1] = states[1:]
    next_states[-1] = states[-1]
    terminal_idx = np.where(terminals > 0.5)[0]
    next_states[terminal_idx] = states[terminal_idx]

    # Save
    np.savez_compressed(
        args.out_npz,
        observations=states,
        actions=actions,
        rewards=rewards,
        next_observations=next_states,
        terminals=terminals,
        episode_index=ep.astype(np.int32),
        frame_index=fr.astype(np.int32),
        ds_index=ds_indices,
        # Debug outputs
        r_left=r_left,
        r_right=r_right,
        success_mask=success_mask.astype(np.uint8),
    )

    print(f"Saved: {args.out_npz}")
    print(f"Shapes: S={states.shape}, A={actions.shape}, R={rewards.shape}")

    # ======================== Verification ========================
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    # 1. State dimension check
    expected_dim = front_dim + left_dim + right_dim + obs_state.shape[1]
    print(f"\n[1] State Dimension Check:")
    print(f"    Front latent: {front_dim}d")
    print(f"    Left latent: {left_dim}d")
    print(f"    Right latent: {right_dim}d")
    print(f"    Observation state: {obs_state.shape[1]}d")
    print(f"    Expected total: {expected_dim}d")
    print(f"    Actual: {states.shape[1]}d")
    assert states.shape[1] == expected_dim, f"State dim mismatch!"
    print(f"    -> OK")

    # 2. Episode/timestep alignment check
    print(f"\n[2] Episode/Timestep Alignment Check:")

    # Check: where done=0, next row should be same episode and frame+1
    idx_cont = np.where(terminals[:-1] < 0.5)[0]
    ep_match = np.all(ep[idx_cont] == ep[idx_cont + 1])
    fr_match = np.all(fr[idx_cont + 1] == fr[idx_cont] + 1)
    print(f"    Continuous transitions (done=0): {len(idx_cont)}")
    print(f"    Episode continues correctly: {ep_match}")
    print(f"    Frame index increments by 1: {fr_match}")
    assert ep_match, "Found done=0 but episode changes!"
    assert fr_match, "Found done=0 but frame_index not consecutive!"

    # Check: where done=1, next row should NOT be valid continuation
    idx_term = np.where(terminals[:-1] > 0.5)[0]
    if len(idx_term) > 0:
        bad = np.where((ep[idx_term] == ep[idx_term + 1]) & (fr[idx_term + 1] == fr[idx_term] + 1))[0]
        print(f"    Terminal transitions (done=1): {len(idx_term)}")
        print(f"    Invalid continuations after terminal: {len(bad)}")
        assert len(bad) == 0, "Found done=1 but next is still a valid continuation!"

    print(f"    -> OK: Episode/timestep alignment is consistent.")

    # 3. Summary statistics
    print(f"\n[3] Summary Statistics:")
    print(f"    Total transitions: {N}")
    print(f"    Number of episodes: {len(np.unique(ep))}")
    print(f"    Terminal states: {int(terminals.sum())}")
    print(f"    Success frames: {success_mask.sum()} ({success_mask.mean()*100:.2f}%)")
    print(f"    Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    print(f"    Reward mean: {rewards.mean():.4f}")

    print("\n" + "="*60)
    print("ALL CHECKS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
