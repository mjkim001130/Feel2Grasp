"""
SO101 Robot Evaluation with IQL Policy (Frame Stacked)

This version uses frame stacking (4 frames of 72D = 288D) for temporal context.
Based on so101_iql_eval_smooth.py with frame stacking support.

Usage:
    python so101_iql_eval_stacked.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from circle_detector import CircleDetector


# ======================== Configuration ========================
# IQL Checkpoint paths (update these after training)
IQL_CHECKPOINT_PATH = "IQL_checkpoints/iql_stacked4_step_2000000_0_0.7_3.0.pt"
# IQL_CHECKPOINT_PATH = "IQL_checkpoints/iql_stacked4_step_1500000_0_0.7_3.0.pt"
ENCODER_CHECKPOINT_PATH = "ae_out/encoder.pt"

# Frame stacking
STACK_SIZE = 4

# Episode configuration
NUM_EPISODES = 1
FPS = 25
EPISODE_TIME_SEC = 60
DISPLAY_DATA = True
DEVICE = "cuda"

# Action speed control
MAX_JOINT_SPEED = 7

# Action smoothing (adaptive)
# Base smoothing: used when action change is small
# Min smoothing: used when action change is large (for faster response)
ACTION_SMOOTHING_BASE = 0.35  # smoothing for large changes (more smoothing)
ACTION_SMOOTHING_MAX = 0.9   
ACTION_CHANGE_THRESHOLD = 0.5 

# Circle detection
NUM_BG_FRAMES = 8
K_STD = 1.5


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
        mu, _ = self(s)
        return mu


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


# ======================== Frame Stacked Policy ========================
class IQLFrameStackedPolicy:
    """
    IQL Policy with frame stacking for temporal context.

    Maintains a buffer of recent 72D observations and stacks them
    to create 288D (72 x 4) input for the policy network.
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder_path: str,
        circle_detector=None,
        device: str = "cuda",
        stack_size: int = 4,
        image_resize_hw: tuple = (128, 128),
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.circle_detector = circle_detector
        self.stack_size = stack_size
        self.image_resize_hw = image_resize_hw
        self.base_obs_dim = 72  # front_enc(64) + circle_det(2) + joints(6)

        # Load IQL checkpoint
        print(f"Loading IQL checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.obs_dim = ckpt["obs_dim"]
        self.act_dim = ckpt["act_dim"]
        hidden = ckpt.get("hidden", 256)

        # Verify dimensions
        checkpoint_stack_size = ckpt.get("stack_size", stack_size)
        if checkpoint_stack_size != stack_size:
            print(f"Warning: checkpoint stack_size={checkpoint_stack_size}, using {stack_size}")
            self.stack_size = checkpoint_stack_size

        expected_dim = self.base_obs_dim * self.stack_size
        if self.obs_dim != expected_dim:
            print(f"Warning: obs_dim={self.obs_dim}, expected {expected_dim}")
            if self.obs_dim % 72 == 0:
                self.stack_size = self.obs_dim // 72
                print(f"  Adjusted stack_size to {self.stack_size}")

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

        # Freeze parameters
        for p in self.policy_net.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=self.stack_size)
        self.reset()

        print(f"IQL Policy (Frame Stacked) loaded!")
        print(f"  obs_dim={self.obs_dim}, act_dim={self.act_dim}, stack_size={self.stack_size}")

    def reset(self):
        """Reset frame buffer with zeros."""
        self.frame_buffer.clear()
        for _ in range(self.stack_size):
            self.frame_buffer.append(np.zeros(self.base_obs_dim, dtype=np.float32))
        print("[Policy] Frame buffer reset")

    @torch.no_grad()
    def select_action(self, obs_dict: dict, left_key: str = None, right_key: str = None) -> np.ndarray:
        """
        Select action given observation dictionary.

        Args:
            obs_dict: Dictionary with observation keys (no batch dimension)
            left_key: Key for left camera image
            right_key: Key for right camera image

        Returns:
            action: (6,) numpy array
        """
        # 1. Encode front image
        front_img = obs_dict.get('front', None)
        if front_img is None:
            raise KeyError("Front image not found in observation")

        img_latent = self._encode_image(front_img)  # (64,)

        # 2. Get circle detection
        circle_det = np.zeros(2, dtype=np.float32)
        if self.circle_detector is not None and left_key and right_key:
            left_img = obs_dict.get(left_key)
            right_img = obs_dict.get(right_key)
            if left_img is not None and right_img is not None:
                left_circle = self.circle_detector.detect(left_img, "left")
                right_circle = self.circle_detector.detect(right_img, "right")
                circle_det = np.array([float(left_circle), float(right_circle)], dtype=np.float32)

        # 3. Get joint positions
        joint_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                       'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        joints = np.array([obs_dict[k] for k in joint_names], dtype=np.float32)

        # 4. Build current 72D observation
        current_obs = np.concatenate([img_latent, circle_det, joints])  # (72,)

        # 5. Update frame buffer (shift left, add new at end)
        self.frame_buffer.append(current_obs)

        # 6. Stack frames: [oldest, ..., newest] -> flatten
        stacked_obs = np.array(list(self.frame_buffer)).flatten()  # (288,)

        # 7. Get action from policy
        state = torch.from_numpy(stacked_obs).float().unsqueeze(0).to(self.device)  # (1, 288)
        action = self.policy_net.get_action(state)  # (1, 6)

        return action[0].cpu().numpy()

    def _encode_image(self, img: np.ndarray) -> np.ndarray:
        """Encode single image to 64D latent."""
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
        img = F.interpolate(img, size=self.image_resize_hw, mode="bilinear", align_corners=False)

        # Encode
        latent = self.encoder(img)  # (1, 64)

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

    # ======================== Initialize Circle Detector ========================
    log_say("Initializing circle detector...")
    log_say(f"Collecting {NUM_BG_FRAMES} background frames - keep robot stationary!")

    circle_detector = CircleDetector(num_bg_frames=NUM_BG_FRAMES, k_std=K_STD, use_center_crop=True, history_size=10)

    obs = robot.get_observation()
    print(f"Available observation keys: {list(obs.keys())}")

    left_key = None
    right_key = None
    for key in obs.keys():
        key_lower = key.lower()
        if 'left' in key_lower and not key_lower.endswith('.pos'):
            left_key = key
        elif 'right' in key_lower and not key_lower.endswith('.pos'):
            right_key = key

    if left_key is None or right_key is None:
        log_say("Warning: Could not find left/right image keys. Circle detection disabled.")
        circle_detector = None
    else:
        print(f"Using keys: left={left_key}, right={right_key}")

        import cv2
        for i in range(NUM_BG_FRAMES):
            obs = robot.get_observation()
            left_img = obs[left_key]
            right_img = obs[right_key]

            left_img_resized = cv2.resize(left_img, (640, 480))
            right_img_resized = cv2.resize(right_img, (640, 480))

            circle_detector.add_background_frame(left_img_resized, "left")
            circle_detector.add_background_frame(right_img_resized, "right")

            time.sleep(1.0 / FPS)

        log_say("Circle detector initialized!")
        print(f"Status: {circle_detector.get_status()}")

    # ======================== Load IQL Policy ========================
    log_say("Loading IQL policy (frame stacked)...")
    policy = IQLFrameStackedPolicy(
        checkpoint_path=IQL_CHECKPOINT_PATH,
        encoder_path=ENCODER_CHECKPOINT_PATH,
        circle_detector=circle_detector,
        device=DEVICE,
        stack_size=STACK_SIZE,
    )
    log_say("IQL policy loaded!")

    # ======================== Initialize Visualization ========================
    if DISPLAY_DATA:
        init_rerun(session_name="iql_evaluation_stacked")

    # ======================== Initialize Keyboard Listener ========================
    listener, events = init_keyboard_listener()

    # ======================== Run Episode ========================
    log_say(f"Starting IQL evaluation (frame stacked, {STACK_SIZE} frames)")
    log_say(f"State: 72D x {STACK_SIZE} = {72 * STACK_SIZE}D")
    log_say(f"Speed limit: {MAX_JOINT_SPEED} rad/step")
    log_say("Press 'q' to stop")

    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

    # Print initial state
    obs_init = robot.get_observation()
    print("\n" + "="*80)
    print("Initial Joint Positions:")
    print("="*80)
    for name in joint_names:
        pos = obs_init[f"{name}.pos"]
        print(f"  {name:15s}: {pos:8.4f} rad ({np.degrees(pos):7.2f}deg)")
    print("="*80 + "\n")

    try:
        start_time = time.time()
        step = 0
        max_steps = int(EPISODE_TIME_SEC * FPS)

        # Reset policy frame buffer at episode start
        policy.reset()

        # Initialize previous action for smoothing
        prev_action = None

        while step < max_steps:
            step_start = time.time()

            if events["exit_early"]:
                log_say("Stopped by user (q key)")
                break

            # Get observation
            obs = robot.get_observation()
            obs_dict = obs.copy()

            # Resize left/right images
            import cv2
            if left_key and left_key in obs_dict:
                obs_dict[left_key] = cv2.resize(obs_dict[left_key], (640, 480))
            if right_key and right_key in obs_dict:
                obs_dict[right_key] = cv2.resize(obs_dict[right_key], (640, 480))

            # Get action from policy
            raw_action = policy.select_action(obs_dict, left_key, right_key)

            # Apply adaptive action smoothing
            # Large action change -> more smoothing (ACTION_SMOOTHING_BASE)
            # Small action change -> less smoothing (ACTION_SMOOTHING_MAX)
            if prev_action is None:
                target_action = raw_action.copy()
                adaptive_smoothing = 1.0
            else:
                action_change = np.abs(raw_action - prev_action).max()
                # Interpolate smoothing based on action change
                # Large change -> use BASE (more smoothing), Small change -> use MAX (less smoothing)
                t = np.clip(action_change / ACTION_CHANGE_THRESHOLD, 0.0, 1.0)
                adaptive_smoothing = ACTION_SMOOTHING_MAX - t * (ACTION_SMOOTHING_MAX - ACTION_SMOOTHING_BASE)
                target_action = adaptive_smoothing * raw_action + (1 - adaptive_smoothing) * prev_action
            prev_action = target_action.copy()

            # Clip gripper
            target_action[5] = np.clip(target_action[5], 0.0, 42.0)

            # Debug on first step
            if step == 0:
                print("\n" + "="*80)
                print(f"First Policy Output (adaptive smoothing={adaptive_smoothing:.2f}):")
                print("="*80)
                for i, name in enumerate(joint_names):
                    curr = obs_dict[f"{name}.pos"]
                    raw = raw_action[i]
                    targ = target_action[i]
                    print(f"  {name:15s}: current={curr:8.2f}, raw={raw:8.2f}, smoothed={targ:8.2f}")
                print("="*80 + "\n")

            # Apply speed limit
            action_dict = {}
            for key in obs_dict.keys():
                if key.endswith('.pos'):
                    action_dict[key] = obs_dict[key]

            for i, name in enumerate(joint_names):
                current_pos = obs_dict[f"{name}.pos"]
                target_pos = target_action[i]

                diff = target_pos - current_pos
                diff_limited = np.clip(diff, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
                next_pos = current_pos + diff_limited

                if name == "gripper":
                    next_pos = np.clip(next_pos, 0.0, 42.0)

                action_dict[f"{name}.pos"] = next_pos

            # Send action
            robot.send_action(action_dict)

            # Circle detection for logging
            left_circle = 0
            right_circle = 0
            if circle_detector and left_key and right_key:
                left_circle = circle_detector.detect(obs_dict[left_key], "left")
                right_circle = circle_detector.detect(obs_dict[right_key], "right")

            # Display
            if DISPLAY_DATA and step % 1 == 0:
                log_rerun_data(observation=obs_dict, action=action_dict)

            # Print state every timestep (joint positions only)
            current_joints = np.array([obs_dict[f"{name}.pos"] for name in joint_names])
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
