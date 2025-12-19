#!/usr/bin/env python
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets import load_from_disk
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------
# Helpers: adapted from your circle.py
# ---------------------------------------------------------------------

def tensor_to_rgb_numpy(img):
    """LeRobot image tensor -> uint8 RGB HxWx3."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.array(img)

    # CxHxW -> HxWxC
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)

    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    return arr


def get_gray_frame(ds, idx, cam_key):
    """
    Get grayscale float32 frame from LeRobotDataset at index idx, camera cam_key.
    Returns None if decoding fails (we keep alignment by later using score=0).
    """
    try:
        sample = ds[idx]
        img_t = sample[cam_key]
        rgb = tensor_to_rgb_numpy(img_t)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return gray
    except Exception as e:
        print(f"[WARN] Failed to decode sample {idx}, cam {cam_key}: {e}")
        return None


def build_background_episode(ds, idxs, cam_key, num_bg_frames=30):
    """
    Episode-level background: average first num_bg_frames valid frames (like your build_background).
    """
    frames = []
    for idx in idxs[:num_bg_frames]:
        gray = get_gray_frame(ds, idx, cam_key)
        if gray is None:
            continue
        frames.append(gray)

    if not frames:
        raise RuntimeError(f"No valid frames for background in episode (cam={cam_key})")

    bg = np.mean(frames, axis=0).astype(np.float32)
    return bg


def compute_diff_scores_episode(ds, idxs, cam_key, bg_gray, use_center_crop=True):
    """
    For each frame in this episode, compute mean abs diff from bg_gray,
    exactly like compute_diff_scores in your script.
    """
    scores = []

    for idx in idxs:
        gray = get_gray_frame(ds, idx, cam_key)
        if gray is None:
            scores.append(0.0)
            continue

        diff = np.abs(gray - bg_gray)

        if use_center_crop:
            h, w = diff.shape
            # central 60% region (same idea as your code)
            y0, y1 = int(h * 0.2), int(h * 0.8)
            x0, x1 = int(w * 0.2), int(w * 0.8)
            diff = diff[y0:y1, x0:x1]

        score = float(np.mean(diff))
        scores.append(score)

    return scores


def scores_to_mask(scores, k_std=1.5, min_len=2):
    """
    Very close to your scores_to_mask_and_intervals, but we only return
    the boolean mask (one bool per frame index in this episode).

    k_std: threshold = mean + k_std*std
    min_len: drop 'on' segments shorter than this (in frames).
    """
    scores_arr = np.array(scores, dtype=np.float32)
    if scores_arr.size == 0:
        return [False] * 0, 0.0

    mean = float(scores_arr.mean())
    std = float(scores_arr.std() + 1e-8)
    threshold = mean + k_std * std

    mask = [float(s) >= threshold for s in scores]

    # remove short intervals
    filtered = mask[:]
    start = None
    for i, val in enumerate(mask + [False]):  # sentinel False at end
        if val and start is None:
            start = i
        elif not val and start is not None:
            run_len = i - start
            if run_len < min_len:
                for j in range(start, i):
                    filtered[j] = False
            start = None

    return filtered, threshold


# ---------------------------------------------------------------------
# Main labeling script
# ---------------------------------------------------------------------

def main():
    repo_id = "mjkim00/Feel2Grasp"
    print(f"Loading LeRobotDataset from: {repo_id}")
    ds = LeRobotDataset(repo_id, revision="main", force_cache_sync=True)
    hf_ds = ds.hf_dataset

    n_rows = len(hf_ds)
    print("Number of rows:", n_rows)

    # Group row indices by episode_index so we can run your algorithm per episode.
    episode_indices = hf_ds["episode_index"]
    episode_to_rows = defaultdict(list)
    for row_idx, ep in enumerate(episode_indices):
        episode_to_rows[int(ep)].append(row_idx)

    print("Number of episodes:", len(episode_to_rows))

    # Preallocate columns
    left_image_circle = [0] * n_rows
    right_image_circle = [0] * n_rows
    circle_reward = [0] * n_rows

    cam_left_key = "observation.images.left"
    cam_right_key = "observation.images.right"

    for ep, idxs in tqdm(sorted(episode_to_rows.items()),
                         desc="Processing episodes"):
        row_idxs = idxs  # list of dataset indices for this episode

        # ----- LEFT CAMERA -----
        try:
            bg_left = build_background_episode(ds, row_idxs, cam_left_key, num_bg_frames=30)
            left_scores = compute_diff_scores_episode(ds, row_idxs, cam_left_key, bg_left)
            left_mask, left_thr = scores_to_mask(left_scores, k_std=1.5, min_len=2)
        except Exception as e:
            print(f"[WARN] Episode {ep}: failed left cam processing: {e}")
            left_mask = [False] * len(row_idxs)

        # ----- RIGHT CAMERA -----
        try:
            bg_right = build_background_episode(ds, row_idxs, cam_right_key, num_bg_frames=30)
            right_scores = compute_diff_scores_episode(ds, row_idxs, cam_right_key, bg_right)
            right_mask, right_thr = scores_to_mask(right_scores, k_std=1.5, min_len=2)
        except Exception as e:
            print(f"[WARN] Episode {ep}: failed right cam processing: {e}")
            right_mask = [False] * len(row_idxs)

        # Sanity: lengths must match
        assert len(left_mask) == len(row_idxs)
        assert len(right_mask) == len(row_idxs)

        # Fill per-row labels
        for local_i, row_idx in enumerate(row_idxs):
            L = 1 if left_mask[local_i] else 0
            R = 1 if right_mask[local_i] else 0
            left_image_circle[row_idx] = L
            right_image_circle[row_idx] = R

    # Compute reward from flags:
    # 10 if both cams see circle, 1 if only one does, 0 otherwise
    for i in range(n_rows):
        L = left_image_circle[i]
        R = right_image_circle[i]
        if L == 1 and R == 1:
            circle_reward[i] = 10
        elif L == 1 or R == 1:
            circle_reward[i] = 1
        else:
            circle_reward[i] = -1

    # -----------------------------------------------------------------
    # Add columns to HF dataset
    # -----------------------------------------------------------------

    print("Adding columns to HF dataset...")

    # ------------------------------------------------------------
# Add columns to HF dataset
# ------------------------------------------------------------

    hf_ds = ds.hf_dataset   # re-read, just in case
    for col in ["left_image_circle", "right_image_circle", "circle_reward"]:
        if col in hf_ds.column_names:
            hf_ds = hf_ds.remove_columns(col)

    hf_ds = hf_ds.add_column("left_image_circle", left_image_circle)
    hf_ds = hf_ds.add_column("right_image_circle", right_image_circle)
    hf_ds = hf_ds.add_column("circle_reward", circle_reward)

    # Attach back to LeRobotDataset
    ds.hf_dataset = hf_ds

    # -----------------------------------------------------------------
    # Save locally
    # -----------------------------------------------------------------
    save_dir = "/home/exaflops/Documents/Feel2Grasp_circle_bg"  # change if you like
    os.makedirs(save_dir, exist_ok=True)

    # IMPORTANT: re-read from ds to avoid hf_ds accidentally being None
    hf_ds_to_save = ds.hf_dataset
    assert hf_ds_to_save is not None, "ds.hf_dataset is None, something went wrong."

    print(f"Saving modified dataset to disk: {save_dir}")
    save_dir = "/home/exaflops/Documents/Feel2Grasp_circle_bg"  # or any path
    os.makedirs(save_dir, exist_ok=True)

    # get HF dataset, but drop non-serializable format/transform
    hf_ds_to_save = ds.hf_dataset
    # remove torch format / columns / transform so it's pure Arrow
    hf_ds_to_save = hf_ds_to_save.with_format(type=None)
    hf_ds_to_save.set_transform(None)

    print(f"Saving modified dataset to disk: {save_dir}")
    hf_ds_to_save.save_to_disk(save_dir)
    print("Local save done.")


    # -----------------------------------------------------------------
    # (Optional) Upload to Hugging Face Hub
    # -----------------------------------------------------------------
    # Make sure you're logged in:
    #   huggingface-cli login
    #
    # Then uncomment this:

    # new_repo_id = "exaFLOPs09/Feel2Grasp_circle_bg_v1"
    # print(f"Pushing dataset to HF Hub: {new_repo_id}")
    # hf_ds_to_save.push_to_hub(new_repo_id)
    # print("Upload complete.")

if __name__ == "__main__":
    main()

