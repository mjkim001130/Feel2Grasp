"""
IQL Policy wrapper with Frame Stacking for temporal context.

This policy maintains a buffer of recent observations and stacks them
to provide temporal context, helping distinguish causality between
circle detection and gripper closing.

State: 72D x stack_size (e.g., 72D x 4 = 288D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
from collections import deque


class PreTrainedPolicy(nn.Module):
    """Minimal PreTrainedPolicy interface for compatibility with record_loop."""
    def reset(self):
        """Reset policy state."""
        pass


# ======================== IQL Networks ========================
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


# ======================== IQL PreTrained Policy (Frame Stacked) ========================
class IQLPreTrainedPolicyFrameStacked(PreTrainedPolicy):
    """
    IQL Policy wrapper with Frame Stacking for temporal context.

    This policy:
    1. Encodes front image to 64D using ConvEncoder
    2. Gets circle detection (2D) from observation
    3. Gets joint positions (6D)
    4. Builds 72D state and maintains frame buffer
    5. Stacks frames: 72D x stack_size
    6. Runs IQL policy
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder_path: str,
        circle_detector=None,
        config=None,
        device: str = "cuda",
        image_resize_hw: tuple = (128, 128),
        stack_size: int = 4,
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_resize_hw = image_resize_hw
        self.circle_detector = circle_detector
        self._debug_last_step = False
        self.stack_size = stack_size

        # Load IQL checkpoint
        print(f"Loading IQL checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.obs_dim = ckpt["obs_dim"]  # Should be 72 * stack_size
        self.act_dim = ckpt["act_dim"]
        hidden = ckpt.get("hidden", 256)

        # Infer stack size from checkpoint if available
        if "stack_size" in ckpt:
            checkpoint_stack_size = ckpt["stack_size"]
            if checkpoint_stack_size != stack_size:
                print(f"Warning: checkpoint stack_size={checkpoint_stack_size}, but using stack_size={stack_size}")
                self.stack_size = checkpoint_stack_size

        # Verify obs_dim matches expected stacked dimension
        expected_dim = 72 * self.stack_size
        if self.obs_dim != expected_dim:
            print(f"Warning: Expected obs_dim={expected_dim} (72 x {self.stack_size}), got {self.obs_dim}")
            # Try to infer stack size from obs_dim
            if self.obs_dim % 72 == 0:
                inferred_stack = self.obs_dim // 72
                print(f"  Inferring stack_size={inferred_stack} from obs_dim")
                self.stack_size = inferred_stack

        self.base_obs_dim = 72  # Base observation dimension before stacking

        # Load policy network
        hidden_dims = (hidden, hidden)
        self.policy_net = GaussianPolicy(self.obs_dim, self.act_dim, hidden_dims)
        self.policy_net.load_state_dict(ckpt["pi"])
        self.policy_net.to(self.device)
        self.policy_net.eval()

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

        # Initialize frame buffer (deque for efficient FIFO)
        # Buffer stores 72D vectors: [img_enc(64) + circle_det(2) + joints(6)]
        self.frame_buffer = deque(maxlen=self.stack_size)
        self._init_frame_buffer()

        print(f"IQL Policy (Frame Stacked) loaded!")
        print(f"  obs_dim={self.obs_dim}, act_dim={self.act_dim}, stack_size={self.stack_size}")

    def _init_frame_buffer(self):
        """Initialize frame buffer with zeros."""
        self.frame_buffer.clear()
        for _ in range(self.stack_size):
            self.frame_buffer.append(np.zeros(self.base_obs_dim, dtype=np.float32))

    def reset(self):
        """Reset policy state (clear frame buffer)."""
        self._init_frame_buffer()
        print("[Policy] Frame buffer reset")

    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Forward pass (same as select_action for IQL)."""
        return self.select_action(observation)

    def predict_action_chunk(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Predict action chunk (IQL predicts single action, not chunk)."""
        action = self.select_action(observation)
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
        # Find keys dynamically
        front_key = None
        left_key = None
        right_key = None

        for key in observation.keys():
            key_lower = key.lower()
            if 'front' in key_lower:
                front_key = key
            elif 'left' in key_lower and not key_lower.endswith('.pos'):
                left_key = key
            elif 'right' in key_lower and not key_lower.endswith('.pos'):
                right_key = key

        # Get joint positions
        joint_names_ordered = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                               'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        joint_values = [observation[k] for k in joint_names_ordered]
        if isinstance(joint_values[0], torch.Tensor):
            joint_state = torch.stack([v.squeeze(-1) if v.dim() > 1 else v for v in joint_values], dim=-1)
            joint_state = joint_state.cpu().numpy()
        else:
            joint_state = np.stack([v.squeeze(-1) if v.ndim > 1 else v for v in joint_values], axis=-1)

        # Extract front image
        front_img = observation[front_key] if front_key else None
        if front_img is None:
            raise KeyError(f"Front image not found. Available: {list(observation.keys())}")

        batch_size = front_img.shape[0]

        # 1. Encode front image
        img_latent = self._encode_front_image_batch(front_img)  # (B, 64)
        img_latent_np = img_latent.cpu().numpy()

        # 2. Get circle detection
        circle_det = np.zeros((batch_size, 2), dtype=np.float32)
        if self.circle_detector is not None and left_key and right_key:
            for b in range(batch_size):
                left_img = observation[left_key][b]
                right_img = observation[right_key][b]
                if isinstance(left_img, torch.Tensor):
                    left_img = left_img.cpu().numpy()
                    right_img = right_img.cpu().numpy()

                left_circle = self.circle_detector.detect(left_img, "left")
                right_circle = self.circle_detector.detect(right_img, "right")
                circle_det[b] = [float(left_circle), float(right_circle)]

        # 3. Build current 72D observation
        # [img_enc(64) + circle_det(2) + joints(6)] = 72D
        current_obs = np.concatenate([img_latent_np, circle_det, joint_state], axis=1)  # (B, 72)

        # 4. Update frame buffer and get stacked state
        # For simplicity, we handle batch_size=1 case
        # For batch_size > 1, each sample would need its own buffer (not typical in eval)
        if batch_size == 1:
            self.frame_buffer.append(current_obs[0])
            stacked_frames = np.array(list(self.frame_buffer))  # (stack_size, 72)
            stacked_state = stacked_frames.flatten()  # (72 * stack_size,)
            stacked_state = stacked_state[np.newaxis, :]  # (1, 72 * stack_size)
        else:
            # For batched inference, don't use frame buffer (each sample is independent)
            # This is typically only used during training, not eval
            stacked_state = np.tile(current_obs, self.stack_size)  # Simple repeat

        # Convert to tensor
        state = torch.from_numpy(stacked_state).float().to(self.device)

        # Debug: print state info
        if self._debug_last_step:
            print(f"  [Policy Debug] Frame buffer size: {len(self.frame_buffer)}")
            print(f"  [Policy Debug] Current obs (72D): [{', '.join([f'{current_obs[0, i]:.2f}' for i in range(min(10, 72))])}...]")
            print(f"  [Policy Debug] Stacked state shape: {state.shape}")
            print(f"  [Policy Debug] Circle detection: left={circle_det[0, 0]}, right={circle_det[0, 1]}")

        # 5. Get action
        action = self.policy_net.get_action(state)  # (B, 6)

        # Print debug info
        joints_str = ', '.join([f'{joint_state[0, i]:.2f}' for i in range(6)])
        circle_str = f"L={circle_det[0, 0]:.0f}, R={circle_det[0, 1]:.0f}"
        action_str = ', '.join([f'{action[0, i].item():.2f}' for i in range(6)])
        print(f"[State] joints=[{joints_str}], circle=[{circle_str}]")
        print(f"[Action] output: [{action_str}]")

        if self._debug_last_step:
            print("=" * 80)
            self._debug_last_step = False

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
