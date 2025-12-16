#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# -------------------------
# Encoders
# -------------------------
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int, resize_hw=(128, 128), in_ch: int = 3, num_down: int = 4):
        super().__init__()
        self.resize_hw = resize_hw
        self.num_down = num_down

        chs = [32, 64, 128, 256]
        if num_down == 5:
            chs = [32, 64, 128, 256, 256]

        layers = []
        c_in = in_ch
        for c_out in chs:
            layers += [nn.Conv2d(c_in, c_out, 4, 2, 1), nn.ReLU()]
            c_in = c_out
        self.net = nn.Sequential(*layers)

        H, W = resize_hw
        ds = 2 ** num_down
        assert H % ds == 0 and W % ds == 0, f"resize_hw must be divisible by {ds}"
        fc_in = chs[-1] * (H // ds) * (W // ds)
        self.fc = nn.Linear(fc_in, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h)
@dataclass
class EncoderBundle:
    model: nn.Module
    resize_hw: Tuple[int, int]
    latent_dim: int

def load_encoder_pt(path: str, device: torch.device, override_latent_dim: Optional[int] = None) -> EncoderBundle:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["encoder_state_dict"]

    latent_dim = int(ckpt.get("latent_dim", sd["fc.weight"].shape[0]))
    if override_latent_dim is not None:
        latent_dim = int(override_latent_dim)

    # Detect whether checkpoint has the 5th conv (net.8.* exists only in 5-conv version)
    num_down = 5 if ("net.8.weight" in sd or "net.8.bias" in sd) else 4

    resize_hw = tuple(ckpt.get("resize_hw", (128, 128)))

    model = ConvEncoder(latent_dim=latent_dim, resize_hw=resize_hw, num_down=num_down)
    model.load_state_dict(sd, strict=True)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    return EncoderBundle(model=model, resize_hw=resize_hw, latent_dim=latent_dim)





# -------------------------
# Utilities
# -------------------------
def to_float01(img: torch.Tensor) -> torch.Tensor:
    """
    img: (C,H,W), can be uint8 [0..255] or float [0..255] or float [0..1]
    Returns float32 [0..1]
    """
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    return img


@torch.inference_mode()
def encode_images(
    ds: LeRobotDataset,
    ds_indices: np.ndarray,
    key: str,
    enc: EncoderBundle,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Encode per-sample images (decoded frames) into latents.
    Returns: (N, enc.latent_dim) float32
    """
    N = len(ds_indices)
    out = np.zeros((N, enc.latent_dim), dtype=np.float32)

    H, W = enc.resize_hw

    for s in tqdm(range(0, N, batch_size), desc=f"Encode {key}"):
        e = min(N, s + batch_size)
        idx_batch = ds_indices[s:e]

        imgs = []
        for idx in idx_batch:
            sample = ds[int(idx)]
            img = sample[key]  # expected (C,H,W)
            img = to_float01(img)
            imgs.append(img)

        x = torch.stack(imgs, dim=0).to(device)  # (B,C,H,W)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        z = enc.model(x).float().cpu().numpy()
        out[s:e] = z

    return out


def build_ds_index_map(ds: LeRobotDataset) -> Dict[Tuple[int, int], int]:
    """
    Map (episode_index, frame_index) -> ds_index
    Works if ds[i] has metadata fields. If your ds returns these under different names,
    edit this accordingly.
    """
    mapping = {}
    for i in tqdm(range(len(ds)), desc="Build (ep,frame)->ds_index map"):
        sample = ds[i]
        ep = int(sample["episode_index"])
        fr = int(sample["frame_index"])
        mapping[(ep, fr)] = i
    return mapping


def terminals_from_ep_frame(episode_index: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    """
    terminal[t]=1 if next row is not (same episode, frame+1)
    """
    N = len(episode_index)
    term = np.zeros((N,), dtype=np.uint8)
    for t in range(N - 1):
        if (episode_index[t + 1] != episode_index[t]) or (frame_index[t + 1] != frame_index[t] + 1):
            term[t] = 1
    term[N - 1] = 1  # last transition is terminal
    return term


def latent_reward(
    z: np.ndarray,
    ep: np.ndarray,
    base_z_by_ep: Dict[int, np.ndarray],
    circle_z_by_ep: Dict[int, np.ndarray],
    global_circle_z: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    z: (N,D)
    reward per timestep in [-1,1] via relative distance to base vs circle
    """
    N, D = z.shape
    r = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        e = int(ep[i])
        z_base = base_z_by_ep[e]
        z_circle = circle_z_by_ep.get(e, global_circle_z)

        d_base = np.linalg.norm(z[i] - z_base)
        d_circle = np.linalg.norm(z[i] - z_circle)
        r[i] = (d_base - d_circle) / (d_base + d_circle + eps)
    return r


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="LeRobot dataset repo_id")
    ap.add_argument("--parquet", required=True, help="Path to train.parquet")
    ap.add_argument("--out_npz", default="replay_buffer_latent_reward.npz")

    ap.add_argument("--front_encoder_pt", required=True, help="encoder.pt for front latent (e.g., 64d)")
    ap.add_argument("--left_encoder_pt", required=True, help="encoder.pt for left latent (e.g., 8d)")
    ap.add_argument("--right_encoder_pt", required=True, help="encoder.pt for right latent (e.g., 8d)")

    ap.add_argument("--front_key", default="observation.images.front")
    ap.add_argument("--left_key", default="observation.images.left")
    ap.add_argument("--right_key", default="observation.images.right")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--reward_success_value", type=float, default=10.0, help="Value in parquet that denotes circle/success")
    ap.add_argument("--scale_reward_to_10", action="store_true", help="Scale (left+right) reward from ~[-2,2] to [-10,10]")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load encoders
    front_enc = load_encoder_pt(args.front_encoder_pt, device=device)
    left_enc = load_encoder_pt(args.left_encoder_pt, device=device)    # should be 8d
    right_enc = load_encoder_pt(args.right_encoder_pt, device=device)  # should be 8d

    # Load parquet and validate
    df = pd.read_parquet(args.parquet)
    need_cols = ["action", "observation.state", "episode_index", "frame_index"]
    # for circle prototype selection:
    if "circle_reward" not in df.columns:
        raise ValueError("parquet must contain circle_reward so we can define circle/success frames")
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # Arrays from parquet
    episode_index = df["episode_index"].astype(np.int64).to_numpy()
    frame_index = df["frame_index"].astype(np.int64).to_numpy()

    actions = np.stack(df["action"].to_list()).astype(np.float32)
    obs_state = np.stack(df["observation.state"].to_list()).astype(np.float32)

    circle_reward_raw = df["circle_reward"].astype(np.float32).to_numpy()
    success_mask = circle_reward_raw >= float(args.reward_success_value) - 1e-6

    # Load dataset
    ds = LeRobotDataset(args.repo_id, split="train")
    print("Dataset length:", len(ds), "Parquet rows:", len(df))

    # Build ds_index list aligned to parquet order
    # If parquet has "index" that matches ds, you can use it; otherwise map by (episode, frame)
    if "index" in df.columns:
        ds_index = df["index"].astype(np.int64).to_numpy()
        # quick sanity check (optional, comment out if expensive)
        # s = ds[int(ds_index[0])]
        # assert int(s["episode_index"]) == int(episode_index[0]) and int(s["frame_index"]) == int(frame_index[0])
    else:
        mapping = build_ds_index_map(ds)
        ds_index = np.array([mapping[(int(e), int(f))] for e, f in zip(episode_index, frame_index)], dtype=np.int64)

    # Encode latents
    Z_front = encode_images(ds, ds_index, args.front_key, front_enc, device=device, batch_size=args.batch_size)
    Z_left  = encode_images(ds, ds_index, args.left_key,  left_enc,  device=device, batch_size=args.batch_size)
    Z_right = encode_images(ds, ds_index, args.right_key, right_enc, device=device, batch_size=args.batch_size)

    # Build base latent per episode = first frame latent (per camera)
    base_left_by_ep: Dict[int, np.ndarray] = {}
    base_right_by_ep: Dict[int, np.ndarray] = {}
    # also circle prototype per episode (mean over success frames)
    circle_left_by_ep: Dict[int, np.ndarray] = {}
    circle_right_by_ep: Dict[int, np.ndarray] = {}

    # global circle prototype (fallback)
    if success_mask.any():
        global_circle_left = Z_left[success_mask].mean(axis=0)
        global_circle_right = Z_right[success_mask].mean(axis=0)
    else:
        # If you truly have no success labels, you need another way to define circle prototype.
        # Here we just pick dataset mean as a harmless fallback.
        global_circle_left = Z_left.mean(axis=0)
        global_circle_right = Z_right.mean(axis=0)

    # per-episode aggregation
    # iterate contiguous blocks because df is sorted by (episode, frame)
    N = len(df)
    start = 0
    while start < N:
        ep = int(episode_index[start])
        end = start
        while end < N and int(episode_index[end]) == ep:
            end += 1

        # first frame in this episode block
        base_left_by_ep[ep] = Z_left[start].copy()
        base_right_by_ep[ep] = Z_right[start].copy()

        # success frames within this episode
        m = success_mask[start:end]
        if m.any():
            circle_left_by_ep[ep] = Z_left[start:end][m].mean(axis=0)
            circle_right_by_ep[ep] = Z_right[start:end][m].mean(axis=0)

        start = end

    # Compute latent-based reward
    r_left = latent_reward(Z_left, episode_index, base_left_by_ep, circle_left_by_ep, global_circle_left)
    r_right = latent_reward(Z_right, episode_index, base_right_by_ep, circle_right_by_ep, global_circle_right)
    rewards = r_left + r_right  # ~[-2,2]

    if args.scale_reward_to_10:
        rewards = rewards * 5.0  # ~[-10,10]

    # Build replay states
    # state = [front_z, left_z8, right_z8, obs_state]
    states = np.concatenate([Z_front, Z_left, Z_right, obs_state], axis=1).astype(np.float32)

    # terminals + next_states
    terminals = terminals_from_ep_frame(episode_index, frame_index)
    next_states = np.roll(states, shift=-1, axis=0)
    # self-loop terminals
    next_states[terminals.astype(bool)] = states[terminals.astype(bool)]

    # Save
    np.savez_compressed(
        args.out_npz,
        observations=states,
        actions=actions,
        rewards=rewards.astype(np.float32),
        next_observations=next_states,
        terminals=terminals,
        episode_index=episode_index,
        frame_index=frame_index,
        ds_index=ds_index,
        # optional debug outputs
        r_left=r_left,
        r_right=r_right,
        success_mask=success_mask.astype(np.uint8),
    )
    print("Saved:", args.out_npz)
    print("State dim:", states.shape[1], "Action dim:", actions.shape[1])
    print("Reward stats:", float(rewards.min()), float(rewards.mean()), float(rewards.max()))


if __name__ == "__main__":
    main()
