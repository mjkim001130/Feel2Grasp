"""
IQL Policy wrapper compatible with LeRobot's PreTrainedPolicy interface.
NO CIRCLE DETECTION VERSION - 70D state (64 image + 6 joints)

This module wraps the trained IQL policy to work with LeRobot's record_loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any


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


# ======================== IQL PreTrained Policy (No Circle Detection) ========================
class IQLPreTrainedPolicyNoCircle(PreTrainedPolicy):
    """
    IQL Policy wrapper compatible with LeRobot's PreTrainedPolicy interface.
    NO CIRCLE DETECTION VERSION.

    This policy:
    1. Encodes front image to 64D using ConvEncoder
    2. Concatenates: [encoding(64) + joints(6)] = 70D
    3. Runs IQL policy
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder_path: str,
        config=None,
        device: str = "cuda",
        image_resize_hw: tuple = (128, 128),
    ):
        # Initialize base class
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_resize_hw = image_resize_hw
        self._debug_last_step = False  # For debugging

        # Load IQL checkpoint
        print(f"Loading IQL checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.obs_dim = ckpt["obs_dim"]
        self.act_dim = ckpt["act_dim"]
        hidden = ckpt.get("hidden", 256)

        # Verify obs_dim is 70 (no circle detection)
        if self.obs_dim != 70:
            print(f"Warning: Expected obs_dim=70, got {self.obs_dim}")

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

        print(f"IQL Policy (No Circle) loaded! obs_dim={self.obs_dim}, act_dim={self.act_dim}")

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
        state_key = None

        for key in observation.keys():
            key_lower = key.lower()
            # Look for front camera key
            if 'front' in key_lower:
                front_key = key
            # State is constructed from joint positions
            elif key_lower.endswith('.pos'):
                pass

        # If state_key not found, try to construct from joint positions
        if state_key is None:
            pos_keys = [k for k in observation.keys() if k.endswith('.pos')]
            if len(pos_keys) == 6:
                state_key = 'observation.state'

        # Extract front image
        front_img = observation[front_key] if front_key else None

        # Handle joint state
        if state_key == 'observation.state':
            joint_names_ordered = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                                   'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
            joint_values = [observation[k] for k in joint_names_ordered]
            if isinstance(joint_values[0], torch.Tensor):
                joint_state = torch.stack([v.squeeze(-1) if v.dim() > 1 else v for v in joint_values], dim=-1)
            else:
                joint_state = np.stack([v.squeeze(-1) if v.ndim > 1 else v for v in joint_values], axis=-1)
        elif state_key:
            joint_state = observation[state_key]
        else:
            joint_state = None

        if front_img is None or joint_state is None:
            raise KeyError(f"Required keys not found. Available: {list(observation.keys())}")

        batch_size = front_img.shape[0]

        # 1. Encode front image
        img_latent = self._encode_front_image_batch(front_img)  # (B, 64)

        # 2. Build state (NO circle detection)
        if isinstance(joint_state, torch.Tensor):
            joint_vec = joint_state.float().to(self.device)
        else:
            joint_vec = torch.from_numpy(joint_state).float().to(self.device)

        # Concatenate: [64 + 6] = 70 (NO circle detection)
        state = torch.cat([img_latent, joint_vec], dim=1)  # (B, 70)

        # Print last 6 dimensions of state every step (joints only)
        last_6 = state[0, -6:].cpu().numpy()  # (6,)
        print(f"[State] last 6 dims (joints): [{', '.join([f'{v:.4f}' for v in last_6])}]")

        # 3. Get action
        action = self.policy_net.get_action(state)  # (B, 6)

        # Debug logging
        if hasattr(self, '_debug_last_step') and self._debug_last_step:
            print("  [Policy Debug] Front image encoding:")
            print(f"    mean={img_latent[0].mean().item():.4f}, std={img_latent[0].std().item():.4f}")
            print(f"  [Policy Debug] Joint positions: [{', '.join([f'{joint_vec[0, i].item():.2f}' for i in range(6)])}]")
            print(f"  [Policy Debug] State (70D): mean={state[0].mean().item():.4f}, std={state[0].std().item():.4f}")
            print(f"  [Policy Debug] Action (6D): [{', '.join([f'{action[0, i].item():.2f}' for i in range(6)])}]")
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
