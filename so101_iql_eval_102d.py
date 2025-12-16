#!/usr/bin/env python3
"""
SO101 Robot Evaluation with IQL Policy (102d state)

State: [front_z(64), left_z(16), right_z(16), obs_state(6)] = 102d
Uses latent encoders for front, left, right cameras.

Usage:
    python so101_iql_eval_102d.py
"""

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# ======================== Configuration ========================
IQL_CHECKPOINT_PATH = "IQL_checkpoints_102d/iql_102d_step_700000_0_0.7_3.0.pt"

FRONT_ENCODER_PATH = "ae_out/encoder.pt"
LEFT_ENCODER_PATH = "ae_side_out/left_encoder.pt"
RIGHT_ENCODER_PATH = "ae_side_out/right_encoder.pt"

# Episode configuration
NUM_EPISODES = 1
FPS = 25
EPISODE_TIME_SEC = 60
DISPLAY_DATA = True
DEVICE = "cuda"

# Action speed control
MAX_JOINT_SPEED = 7

# Action smoothing (adaptive)
ACTION_SMOOTHING_BASE = 0.35
ACTION_SMOOTHING_MAX = 0.9
ACTION_CHANGE_THRESHOLD = 0.5


# ======================== Networks ========================
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
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else (),
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
        mu, _ = self(s)
        return mu


class ConvEncoder(nn.Module):
    """Convolutional encoder for camera images."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def load_encoder(path: str, device: torch.device):
    """Load encoder checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    latent_dim = ckpt["latent_dim"]
    resize_hw = tuple(ckpt.get("resize_hw", (128, 128)))

    model = ConvEncoder(latent_dim)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, resize_hw, latent_dim


# ======================== 102d Policy ========================
class IQL102dPolicy:
    """
    IQL Policy with 102d state.

    State: [front_z(64), left_z(16), right_z(16), obs_state(6)] = 102d
    """

    def __init__(
        self,
        checkpoint_path: str,
        front_encoder_path: str,
        left_encoder_path: str,
        right_encoder_path: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load encoders
        print(f"Loading front encoder from {front_encoder_path}")
        self.front_encoder, self.front_hw, self.front_dim = load_encoder(front_encoder_path, self.device)

        print(f"Loading left encoder from {left_encoder_path}")
        self.left_encoder, self.left_hw, self.left_dim = load_encoder(left_encoder_path, self.device)

        print(f"Loading right encoder from {right_encoder_path}")
        self.right_encoder, self.right_hw, self.right_dim = load_encoder(right_encoder_path, self.device)

        print(f"Encoder dims: front={self.front_dim}, left={self.left_dim}, right={self.right_dim}")

        # Load IQL checkpoint
        print(f"Loading IQL checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.obs_dim = ckpt["obs_dim"]
        self.act_dim = ckpt["act_dim"]
        hidden = ckpt.get("hidden", 256)

        # Verify dimensions
        expected_dim = self.front_dim + self.left_dim + self.right_dim + 6  # +6 for obs_state
        if self.obs_dim != expected_dim:
            print(f"Warning: checkpoint obs_dim={self.obs_dim}, expected={expected_dim}")

        # Load policy network
        hidden_dims = (hidden, hidden)
        self.policy_net = GaussianPolicy(self.obs_dim, self.act_dim, hidden_dims)
        self.policy_net.load_state_dict(ckpt["pi"])
        self.policy_net.to(self.device)
        self.policy_net.eval()

        for p in self.policy_net.parameters():
            p.requires_grad_(False)

        print(f"IQL Policy (102d) loaded!")
        print(f"  obs_dim={self.obs_dim}, act_dim={self.act_dim}")

    @torch.no_grad()
    def select_action(self, obs_dict: dict) -> np.ndarray:
        """
        Select action given observation dictionary.

        Args:
            obs_dict: Dictionary with observation keys

        Returns:
            action: (6,) numpy array
        """
        # 1. Encode front image
        front_img = obs_dict.get("front", None)
        if front_img is None:
            raise KeyError("Front image not found in observation")
        front_z = self._encode_image(front_img, self.front_encoder, self.front_hw)

        # 2. Encode left image
        left_img = obs_dict.get("left", None)
        if left_img is None:
            raise KeyError("Left image not found in observation")
        left_z = self._encode_image(left_img, self.left_encoder, self.left_hw)

        # 3. Encode right image
        right_img = obs_dict.get("right", None)
        if right_img is None:
            raise KeyError("Right image not found in observation")
        right_z = self._encode_image(right_img, self.right_encoder, self.right_hw)

        # 4. Get joint positions
        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        joints = np.array([obs_dict[k] for k in joint_names], dtype=np.float32)

        # 5. Build 102d state: [front_z, left_z, right_z, joints]
        state = np.concatenate([front_z, left_z, right_z, joints])

        # 6. Get action from policy
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.policy_net.get_action(state_t)

        return action[0].cpu().numpy()

    def _encode_image(self, img: np.ndarray, encoder: nn.Module, resize_hw: tuple) -> np.ndarray:
        """Encode single image to latent."""
        # Ensure correct format
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.5:
            img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        # Resize
        img = F.interpolate(img, size=resize_hw, mode="bilinear", align_corners=False)

        # Encode
        latent = encoder(img)

        return latent[0].cpu().numpy()


def main():
    init_logging()

    # ======================== Robot Configuration ========================
    camera_config = {
        "front": OpenCVCameraConfig(index_or_path="/dev/camera_front", width=640, height=480, fps=FPS),
        "left": OpenCVCameraConfig(index_or_path="/dev/camera_left", width=3280, height=2464, fps=FPS),
        "right": OpenCVCameraConfig(index_or_path="/dev/camera_right", width=3280, height=2464, fps=FPS),
    }

    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM1",
        id="black_follower_arm",
        cameras=camera_config,
    )

    # ======================== Initialize Robot ========================
    log_say("Connecting to robot...")
    robot = SO101Follower(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot!")

    log_say("Robot connected!")

    # ======================== Load IQL Policy ========================
    log_say("Loading IQL policy (102d)...")
    policy = IQL102dPolicy(
        checkpoint_path=IQL_CHECKPOINT_PATH,
        front_encoder_path=FRONT_ENCODER_PATH,
        left_encoder_path=LEFT_ENCODER_PATH,
        right_encoder_path=RIGHT_ENCODER_PATH,
        device=DEVICE,
    )
    log_say("IQL policy loaded!")

    # ======================== Initialize Visualization ========================
    if DISPLAY_DATA:
        init_rerun(session_name="iql_evaluation_102d")

    # ======================== Initialize Keyboard Listener ========================
    listener, events = init_keyboard_listener()

    # ======================== Run Episode ========================
    log_say("Starting IQL evaluation (102d state)")
    log_say(f"Speed limit: {MAX_JOINT_SPEED} rad/step")
    log_say("Press 'q' to stop")

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Get observation keys
    obs = robot.get_observation()
    print(f"Available observation keys: {list(obs.keys())}")

    # Find camera keys
    front_key = None
    left_key = None
    right_key = None
    for key in obs.keys():
        key_lower = key.lower()
        if "front" in key_lower and not key_lower.endswith(".pos"):
            front_key = key
        elif "left" in key_lower and not key_lower.endswith(".pos"):
            left_key = key
        elif "right" in key_lower and not key_lower.endswith(".pos"):
            right_key = key

    print(f"Using camera keys: front={front_key}, left={left_key}, right={right_key}")

    if front_key is None or left_key is None or right_key is None:
        raise RuntimeError("Could not find all camera keys!")

    # Print initial state
    obs_init = robot.get_observation()
    print("\n" + "=" * 80)
    print("Initial Joint Positions:")
    print("=" * 80)
    for name in joint_names:
        pos = obs_init[f"{name}.pos"]
        print(f"  {name:15s}: {pos:8.4f} rad ({np.degrees(pos):7.2f}deg)")
    print("=" * 80 + "\n")

    try:
        start_time = time.time()
        step = 0
        max_steps = int(EPISODE_TIME_SEC * FPS)

        # Initialize previous action for smoothing
        prev_action = None

        while step < max_steps:
            step_start = time.time()

            if events["exit_early"]:
                log_say("Stopped by user (q key)")
                break

            # Get observation
            obs = robot.get_observation()

            # Build obs_dict for policy (resize left/right images)
            obs_dict = {
                "front": obs[front_key],
                "left": cv2.resize(obs[left_key], (640, 480)),
                "right": cv2.resize(obs[right_key], (640, 480)),
            }
            for name in joint_names:
                obs_dict[f"{name}.pos"] = obs[f"{name}.pos"]

            # Get action from policy
            raw_action = policy.select_action(obs_dict)

            # Apply adaptive action smoothing
            if prev_action is None:
                target_action = raw_action.copy()
                adaptive_smoothing = 1.0
            else:
                action_change = np.abs(raw_action - prev_action).max()
                t = np.clip(action_change / ACTION_CHANGE_THRESHOLD, 0.0, 1.0)
                adaptive_smoothing = ACTION_SMOOTHING_MAX - t * (ACTION_SMOOTHING_MAX - ACTION_SMOOTHING_BASE)
                target_action = adaptive_smoothing * raw_action + (1 - adaptive_smoothing) * prev_action
            prev_action = target_action.copy()

            # Clip gripper
            target_action[5] = np.clip(target_action[5], 0.0, 42.0)

            # Debug on first step
            if step == 0:
                print("\n" + "=" * 80)
                print(f"First Policy Output (adaptive smoothing={adaptive_smoothing:.2f}):")
                print("=" * 80)
                for i, name in enumerate(joint_names):
                    curr = obs_dict[f"{name}.pos"]
                    raw = raw_action[i]
                    targ = target_action[i]
                    print(f"  {name:15s}: current={curr:8.2f}, raw={raw:8.2f}, smoothed={targ:8.2f}")
                print("=" * 80 + "\n")

            # Apply speed limit
            action_dict = {}
            for key in obs.keys():
                if key.endswith(".pos"):
                    action_dict[key] = obs[key]

            for i, name in enumerate(joint_names):
                current_pos = obs[f"{name}.pos"]
                target_pos = target_action[i]

                diff = target_pos - current_pos
                diff_limited = np.clip(diff, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
                next_pos = current_pos + diff_limited

                if name == "gripper":
                    next_pos = np.clip(next_pos, 0.0, 42.0)

                action_dict[f"{name}.pos"] = next_pos

            # Send action
            robot.send_action(action_dict)

            # Display
            if DISPLAY_DATA and step % 0.1 == 0:
                log_rerun_data(observation=obs, action=action_dict)

            # Print state every timestep
            current_joints = np.array([obs[f"{name}.pos"] for name in joint_names])
            print(f"[Step {step:4d}] joints={np.array2string(current_joints, precision=2, separator=', ')}")

            step += 1

            # Maintain FPS
            step_duration = time.time() - step_start
            sleep_time = (1.0 / FPS) - step_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

        elapsed_time = time.time() - start_time
        actual_fps = step / elapsed_time if elapsed_time > 0 else 0

        log_say("Episode complete!")
        log_say(f"Duration: {elapsed_time:.1f}s, Steps: {step}, Avg FPS: {actual_fps:.1f}")

    except KeyboardInterrupt:
        log_say("Interrupted by user")
    except Exception as e:
        log_say(f"Error during episode: {e}")
        import traceback

        traceback.print_exc()

    # ======================== Cleanup ========================
    log_say("Disconnecting robot...")
    robot.disconnect()

    if listener is not None:
        listener.stop()

    log_say("Done!")


if __name__ == "__main__":
    main()
