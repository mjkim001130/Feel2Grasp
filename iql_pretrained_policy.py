#!/usr/bin/env python
"""
IQL Policy wrapper compatible with LeRobot's PreTrainedPolicy interface.

This module wraps the trained IQL policy to work with LeRobot's record_loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from torch import nn


class PreTrainedPolicy(nn.Module):
    """Minimal PreTrainedPolicy interface for compatibility with record_loop."""
    def reset(self):
        """Reset policy state."""
        pass


# ======================== IQL Networks (from train_iql_test.py) ========================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    """IQL actor."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.backbone = MLP(
            obs_dim,
            hidden_dims[-1],
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else ()
        )
        self.mu = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s):
        h = self.backbone(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def get_action(self, s):
        """Return deterministic action (mean) for inference."""
        mu, _ = self(s)
        return mu


# ======================== ConvEncoder ========================
class ConvEncoder(nn.Module):
    """Convolutional encoder for front camera images."""

    def __init__(self, latent_dim: int = 64):
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


# ======================== Circle Detector ========================
import cv2


class SimpleCircleDetector:
    """Simple circle detector using background subtraction."""

    def __init__(self, k_std: float = 1.5, use_center_crop: bool = True):
        self.k_std = k_std
        self.use_center_crop = use_center_crop

        self.bg_frames = {"left": [], "right": []}
        self.bg_model = {"left": None, "right": None}
        self.threshold = {"left": None, "right": None}
        self.is_finalized = {"left": False, "right": False}

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale float32."""
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.ndim == 3:
            if img.shape[0] == 3:  # CHW
                img = np.transpose(img, (1, 2, 0))  # -> HWC

        if img.ndim == 2:
            return img.astype(np.float32)

        if img.dtype == np.uint8 and img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return gray.astype(np.float32)

        raise ValueError(f"Unsupported image format: shape={img.shape}, dtype={img.dtype}")

    def add_background_frame(self, img, camera: str = "left"):
        """Add frame to background collection."""
        gray = self._to_grayscale(img)
        self.bg_frames[camera].append(gray)

    def finalize_background(self, camera: str = None):
        """Compute final background model from collected frames."""
        if camera is None:
            cameras = ["left", "right"]
        else:
            cameras = [camera]

        for cam in cameras:
            if len(self.bg_frames[cam]) == 0:
                raise ValueError(f"No background frames collected for {cam} camera")

            self.bg_model[cam] = np.mean(self.bg_frames[cam], axis=0).astype(np.float32)

            scores = []
            for frame in self.bg_frames[cam]:
                diff = np.abs(frame - self.bg_model[cam])
                if self.use_center_crop:
                    h, w = diff.shape
                    y0, y1 = int(h * 0.2), int(h * 0.8)
                    x0, x1 = int(w * 0.2), int(w * 0.8)
                    diff = diff[y0:y1, x0:x1]
                scores.append(float(np.mean(diff)))

            mean_score = np.mean(scores)
            std_score = np.std(scores) + 1e-8
            self.threshold[cam] = mean_score + self.k_std * std_score

            self.is_finalized[cam] = True
            print(f"[{cam}] Background finalized: threshold={self.threshold[cam]:.2f}")

    def detect(self, img, camera: str = "left") -> int:
        """Detect circle in current frame."""
        if not self.is_finalized[camera]:
            return 0

        gray = self._to_grayscale(img)
        bg = self.bg_model[camera]

        diff = np.abs(gray - bg)

        if self.use_center_crop:
            h, w = diff.shape
            y0, y1 = int(h * 0.2), int(h * 0.8)
            x0, x1 = int(w * 0.2), int(w * 0.8)
            diff = diff[y0:y1, x0:x1]

        score = float(np.mean(diff))

        return 1 if score >= self.threshold[camera] else 0


# ======================== IQL PreTrained Policy ========================
class IQLPreTrainedPolicy(PreTrainedPolicy):
    """
    IQL Policy wrapper compatible with LeRobot's PreTrainedPolicy interface.

    This policy:
    1. Encodes front image to 64D using ConvEncoder
    2. Detects circles in left/right images (0 or 1)
    3. Concatenates: [encoding(64) + circles(2) + joints(6)] = 72D
    4. Normalizes state
    5. Runs IQL policy
    6. Denormalizes action
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder_path: str,
        circle_detector: SimpleCircleDetector,
        config = None,
        device: str = "cuda",
        image_resize_hw: tuple = (128, 128),
    ):
        # Initialize base class
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_resize_hw = image_resize_hw
        self.circle_detector = circle_detector
        self._debug_last_step = False  # For debugging

        # Load IQL checkpoint
        print(f"Loading IQL checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.obs_dim = ckpt["obs_dim"]
        self.act_dim = ckpt["act_dim"]
        hidden = ckpt.get("hidden", 256)

        # Load policy network
        hidden_dims = (hidden, hidden)
        self.policy_net = GaussianPolicy(self.obs_dim, self.act_dim, hidden_dims)
        self.policy_net.load_state_dict(ckpt["pi"])
        self.policy_net.to(self.device)
        self.policy_net.eval()

        # No normalization - using raw data
        self.use_normalization = False

        # Load image encoder
        print(f"Loading image encoder from {encoder_path}")
        enc_ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
        self.encoder = ConvEncoder(enc_ckpt["latent_dim"])
        self.encoder.load_state_dict(enc_ckpt["encoder_state_dict"])
        self.encoder.to(self.device)
        self.encoder.eval()

        # Freeze all parameters
        for p in self.policy_net.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        print(f"IQL Policy loaded! obs_dim={self.obs_dim}, act_dim={self.act_dim}")

    def reset(self):
        """Reset policy state (IQL is stateless)."""
        pass

    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Forward pass (same as select_action for IQL)."""
        return self.select_action(observation)

    def predict_action_chunk(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Predict action chunk (IQL predicts single action, not chunk)."""
        action = self.select_action(observation)
        # Return with temporal dimension (B, 1, action_dim)
        return action.unsqueeze(1)

    def get_optim_params(self):
        """Get optimizer parameters (not used during inference)."""
        return self.parameters()

    @torch.no_grad()
    def select_action(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Select action given observation batch.

        Args:
            observation: Dict with observation keys

        Returns:
            action: (B, 6) tensor
        """
        # Find image keys dynamically
        front_key = None
        left_key = None
        right_key = None
        state_key = None

        for key in observation.keys():
            key_lower = key.lower()
            # Look for camera keys (may not contain 'image')
            if 'front' in key_lower:
                front_key = key
            elif 'left' in key_lower:
                left_key = key
            elif 'right' in key_lower:
                right_key = key
            # State is constructed from joint positions
            elif key_lower.endswith('.pos'):
                # Will collect all .pos keys later
                pass

        # If state_key not found, try to construct from joint positions
        if state_key is None:
            # Collect all .pos keys
            pos_keys = [k for k in observation.keys() if k.endswith('.pos')]
            if len(pos_keys) == 6:
                # Construct state from joint positions
                state_key = 'observation.state'  # Will handle specially

        # Extract images and state
        front_img = observation[front_key] if front_key else None
        left_img = observation[left_key] if left_key else None
        right_img = observation[right_key] if right_key else None

        # Handle joint state
        if state_key == 'observation.state':
            # Construct from .pos keys in the CORRECT order
            # Training data uses: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
            joint_names_ordered = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                                   'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
            joint_values = [observation[k] for k in joint_names_ordered]
            # Stack into (B, 6)
            if isinstance(joint_values[0], torch.Tensor):
                # Each joint_values[i] is (B, 1), squeeze and stack
                joint_state = torch.stack([v.squeeze(-1) if v.dim() > 1 else v for v in joint_values], dim=-1)
            else:
                # Each joint_values[i] is (B, 1), squeeze and stack
                joint_state = np.stack([v.squeeze(-1) if v.ndim > 1 else v for v in joint_values], axis=-1)
        elif state_key:
            joint_state = observation[state_key]
        else:
            joint_state = None

        if front_img is None or joint_state is None:
            raise KeyError(f"Required keys not found. Available: {list(observation.keys())}")
        if left_img is None or right_img is None:
            print("Warning: Left/Right images not found, circle detection disabled")
            left_img = front_img  # Dummy
            right_img = front_img  # Dummy

        batch_size = front_img.shape[0]

        # 1. Encode front image
        img_latent = self._encode_front_image_batch(front_img)  # (B, 64)

        # 2. Detect circles
        circles = self._detect_circles_batch(left_img, right_img)  # (B, 2)

        # 3. Build state
        if isinstance(joint_state, torch.Tensor):
            joint_vec = joint_state.float().to(self.device)
        else:
            joint_vec = torch.from_numpy(joint_state).float().to(self.device)

        # Concatenate: [64 + 2 + 6] = 72
        state = torch.cat([img_latent, circles, joint_vec], dim=1)  # (B, 72)

        # Print last 8 dimensions of state every step (circle detection + joints)
        last_8 = state[0, -8:].cpu().numpy()  # (8,)
        print(f"[State] last 8 dims: [{', '.join([f'{v:.4f}' for v in last_8])}]")

        # 4. Get action (no normalization)
        action = self.policy_net.get_action(state)  # (B, 6)

        # Debug logging (only on first step)
        if hasattr(self, '_debug_last_step') and self._debug_last_step:
            print("  [Policy Debug] Front image encoding:")
            print(f"    mean={img_latent[0].mean().item():.4f}, std={img_latent[0].std().item():.4f}, min={img_latent[0].min().item():.4f}, max={img_latent[0].max().item():.4f}")
            print(f"  [Policy Debug] Circle detection: left={circles[0, 0].item():.0f}, right={circles[0, 1].item():.0f}")
            print(f"  [Policy Debug] Joint positions: [{', '.join([f'{joint_vec[0, i].item():.2f}' for i in range(6)])}]")
            print(f"  [Policy Debug] State (72D): mean={state[0].mean().item():.4f}, std={state[0].std().item():.4f}")
            print(f"  [Policy Debug] Action (6D): [{', '.join([f'{action[0, i].item():.2f}' for i in range(6)])}]")
            print("=" * 80)
            self._debug_last_step = False  # Reset flag

        # 7. DO NOT wrap angles - policy was trained with unwrapped angles (±100 radians)
        # Wrapping would map trained actions incorrectly (e.g., 98 rad → -3.1 rad)

        return action

    def _encode_front_image_batch(self, front_images: torch.Tensor) -> torch.Tensor:
        """Encode batch of front images to 64D latent."""
        imgs = front_images

        # Ensure tensor
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.from_numpy(imgs)

        # Normalize to [0, 1]
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        else:
            imgs = imgs.float()
            if imgs.max() > 1.5:
                imgs = imgs / 255.0

        # Handle HWC -> CHW
        if imgs.ndim == 4 and imgs.shape[-1] == 3:  # (B, H, W, C)
            imgs = imgs.permute(0, 3, 1, 2)  # -> (B, C, H, W)

        # Move to device and resize
        imgs = imgs.to(self.device)
        imgs = F.interpolate(imgs, size=self.image_resize_hw, mode="bilinear", align_corners=False)

        # Encode
        latent = self.encoder(imgs)  # (B, 64)

        return latent

    def _detect_circles_batch(self, left_images: torch.Tensor, right_images: torch.Tensor) -> torch.Tensor:
        """Detect circles in batch of left/right images."""
        batch_size = left_images.shape[0]

        # If circle detector is not available or not initialized, return zeros
        # CircleDetector uses 'is_initialized', SimpleCircleDetector uses 'is_finalized'
        if self.circle_detector is None:
            return torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)

        # Check initialization status based on detector type
        if hasattr(self.circle_detector, 'is_finalized'):
            # SimpleCircleDetector
            if not self.circle_detector.is_finalized.get("left", False):
                return torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)
        elif hasattr(self.circle_detector, 'is_initialized'):
            # CircleDetector
            if not self.circle_detector.is_initialized.get("left", False):
                return torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)
        else:
            # Unknown detector type, return zeros
            return torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)

        left_circles = []
        right_circles = []

        for i in range(batch_size):
            left_img = left_images[i]
            right_img = right_images[i]

            # Convert to numpy
            if isinstance(left_img, torch.Tensor):
                left_img = left_img.cpu().numpy()
            if isinstance(right_img, torch.Tensor):
                right_img = right_img.cpu().numpy()

            # Detect
            left_circle = self.circle_detector.detect(left_img, "left")
            right_circle = self.circle_detector.detect(right_img, "right")

            left_circles.append(left_circle)
            right_circles.append(right_circle)

        # Stack to tensor
        circles = torch.tensor(
            [[l, r] for l, r in zip(left_circles, right_circles)],
            dtype=torch.float32,
            device=self.device
        )  # (B, 2)

        return circles
