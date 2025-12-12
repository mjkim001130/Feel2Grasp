#!/usr/bin/env python
"""
SO101 Robot Evaluation with IQL Policy (Smooth Motion)

This version includes action interpolation for smoother, slower robot motion.
Based on so101_iql_eval.py but adds interpolation between current and target positions.

Usage:
    python so101_iql_eval_smooth.py
"""

import time
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from iql_pretrained_policy import IQLPreTrainedPolicy
from circle_detector import CircleDetector


# ======================== Configuration ========================
# IQL Checkpoint paths
IQL_CHECKPOINT_PATH = "IQL_checkpoints/iql_step_2000000_full_0.pt"
ENCODER_CHECKPOINT_PATH = "ae_out/encoder.pt"

# Episode configuration
NUM_EPISODES = 1  
FPS = 25
EPISODE_TIME_SEC = 60  
DISPLAY_DATA = False  # Disabled to prevent camera lag
DEVICE = "cuda"

# Action speed control (for smooth motion)
MAX_JOINT_SPEED = 10  # Maximum radians per step (balanced for stability)
# Speed examples (actual speed depends on real FPS, not just config):
#   0.02 rad/step @ 17 FPS = 0.34 rad/s = ~19°/s (very slow)
#   0.05 rad/step @ 17 FPS = 0.85 rad/s = ~49°/s (slow)
#   0.1 rad/step @ 17 FPS = 1.7 rad/s = ~97°/s (moderate)
#   0.2 rad/step @ 17 FPS = 3.4 rad/s = ~195°/s (fast - current)

# Circle detection
NUM_BG_FRAMES = 10
K_STD = 15  # Increased from 1.5 to reduce false positives

# Dataset configuration (optional, for saving evaluation data)
SAVE_DATASET = False  
DATASET_REPO_ID = "mjkim00/IQL_eval"


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
    log_say("Initializing circle detector with adaptive thresholding...")
    log_say(f"Collecting {NUM_BG_FRAMES} background frames - keep robot stationary!")

    circle_detector = CircleDetector(num_bg_frames=NUM_BG_FRAMES, k_std=K_STD, use_center_crop=True, history_size=10)

    # Collect background frames
    # First check what keys are available
    obs = robot.get_observation()
    print(f"Available observation keys: {list(obs.keys())}")

    # Try to find image keys (may not contain 'image' in name)
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
        print(f"Available keys: {list(obs.keys())}")
        circle_detector = None
    else:
        print(f"Using keys: left={left_key}, right={right_key}")

        import cv2
        for i in range(NUM_BG_FRAMES):
            obs = robot.get_observation()
            left_img = obs[left_key]
            right_img = obs[right_key]

            # Resize to (640, 480) before adding to background model
            left_img_resized = cv2.resize(left_img, (640, 480))
            right_img_resized = cv2.resize(right_img, (640, 480))

            circle_detector.add_background_frame(left_img_resized, "left")
            circle_detector.add_background_frame(right_img_resized, "right")

            time.sleep(1.0 / FPS)

            if (i + 1) % 10 == 0:
                print(f"  Collected {i + 1}/{NUM_BG_FRAMES} frames")

        log_say("Circle detector initialized with adaptive thresholding!")
        print(f"Status: {circle_detector.get_status()}")

    # ======================== Load IQL Policy ========================
    log_say("Loading IQL policy...")
    policy = IQLPreTrainedPolicy(
        checkpoint_path=IQL_CHECKPOINT_PATH,
        encoder_path=ENCODER_CHECKPOINT_PATH,
        circle_detector=circle_detector,
        device=DEVICE,
    )
    log_say("IQL policy loaded!")

    # ======================== Setup Processors ========================
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # ======================== Setup Dataset (Optional) ========================
    dataset = None
    if SAVE_DATASET:
        log_say("Creating dataset for saving evaluation data...")

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )

        # Adjust image shapes for resized left/right cameras
        dataset_features['observation.images.left']['shape'] = (480, 640, 3)
        dataset_features['observation.images.right']['shape'] = (480, 640, 3)

        dataset = LeRobotDataset.create(
            DATASET_REPO_ID,
            FPS,
            root=None,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_threads=4 * len(robot.cameras),
        )
    else:
        log_say("Dataset saving disabled (SAVE_DATASET=False)")


    # ======================== Initialize Visualization ========================
    if DISPLAY_DATA:
        init_rerun(session_name="iql_evaluation_smooth")

    # ======================== Initialize Keyboard Listener ========================
    listener, events = init_keyboard_listener()

    # ======================== Run Episode ========================
    log_say(f"Starting IQL evaluation episode (max {EPISODE_TIME_SEC}s)")
    log_say(f"Using speed limit: {MAX_JOINT_SPEED} rad/step = {MAX_JOINT_SPEED * FPS:.2f} rad/s = {np.degrees(MAX_JOINT_SPEED * FPS):.1f}°/s")
    log_say("Press 'q' to stop at any time")
    log_say("Circle detection is used as input to policy, not for termination")

    # ======================== Debug: Print Initial Joint Positions ========================
    obs_init = robot.get_observation()
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    print("\n" + "="*80)
    print("Initial Joint Positions:")
    print("="*80)
    for name in joint_names:
        pos = obs_init[f"{name}.pos"]
        print(f"  {name:15s}: {pos:8.4f} rad ({np.degrees(pos):7.2f}°)")
    print("="*80)

    # Debug: Check state dimensions
    print("\nState Dimension Check:")
    print(f"  Front image shape (original): {obs_init['front'].shape if 'front' in obs_init else 'N/A'}")
    if left_key and left_key in obs_init:
        import cv2
        left_resized = cv2.resize(obs_init[left_key], (640, 480))
        right_resized = cv2.resize(obs_init[right_key], (640, 480))
        print(f"  Left image shape (original): {obs_init[left_key].shape}")
        print(f"  Left image shape (resized for policy): {left_resized.shape}")
        print(f"  Right image shape (original): {obs_init[right_key].shape}")
        print(f"  Right image shape (resized for policy): {right_resized.shape}")
    print(f"  Joint positions: {len([k for k in obs_init.keys() if k.endswith('.pos')])} joints")
    print("  Expected state: 64 (front encoding) + 2 (circle detection) + 6 (joints) = 72D")
    print("="*80 + "\n")

    try:
        # Custom control loop
        start_time = time.time()
        step = 0
        max_steps = int(EPISODE_TIME_SEC * FPS)

        while step < max_steps:
            step_start = time.time()

            # Check for keyboard interrupt
            if events["exit_early"]:
                log_say("Stopped by user (q key)")
                break

            # Get observation
            obs = robot.get_observation()

            # DO NOT wrap joint angles - policy was trained with unwrapped angles
            # Training data has angles like ±100 radians, not [-π, π]
            obs_wrapped = obs.copy()

            # Resize left/right images to (480, 640, 3) like in so101_record.py
            import cv2
            if left_key and left_key in obs_wrapped:
                left_img_resized = cv2.resize(obs_wrapped[left_key], (640, 480))
                obs_wrapped[left_key] = left_img_resized
            if right_key and right_key in obs_wrapped:
                right_img_resized = cv2.resize(obs_wrapped[right_key], (640, 480))
                obs_wrapped[right_key] = right_img_resized

            # Prepare observation batch (policy expects batch dimension)
            obs_batch = {}
            for key, value in obs_wrapped.items():
                if isinstance(value, np.ndarray):
                    # Add batch dimension
                    obs_batch[key] = value[np.newaxis, ...]
                elif isinstance(value, (int, float)):
                    obs_batch[key] = np.array([[value]])
                else:
                    obs_batch[key] = value

            # Get action from policy (returns batch)
            action_batch = policy.select_action(obs_batch)

            # Remove batch dimension
            target_action = action_batch[0].cpu().numpy()

            # Debug: Print policy output on first step
            if step == 0:
                print("\n" + "="*80)
                print("First Policy Output (target_action):")
                print("="*80)
                for i, name in enumerate(joint_names):
                    curr = obs_wrapped[f"{name}.pos"]
                    targ = target_action[i]
                    diff = targ - curr
                    print(f"  {name:15s}: current={curr:8.2f} rad, target={targ:8.2f} rad, diff={diff:8.2f} rad")
                print("="*80 + "\n")

                # Additional debug: Get detailed policy internals
                print("\n" + "="*80)
                print("Policy Internal State (Step 0):")
                print("="*80)
                policy._debug_last_step = True  # Signal to policy to print debug info

            # ======================== Apply speed limit for smooth motion ========================
            # Limit the movement to MAX_JOINT_SPEED radians per step
            # Note: Robot uses unwrapped angles (±100 rad), matching training data format
            # Direct subtraction works fine - no wraparound needed

            # IMPORTANT: Initialize action_dict with ALL current joint positions
            # (robot needs all joint commands, not just the ones we want to move)
            action_dict = {}
            for key in obs_wrapped.keys():
                if key.endswith('.pos'):
                    action_dict[key] = obs_wrapped[key]

            # Now update with speed-limited movements
            for i, name in enumerate(joint_names):
                current_pos = obs_wrapped[f"{name}.pos"]
                target_pos = target_action[i]

                # Compute difference (both in unwrapped space)
                diff = target_pos - current_pos

                # Limit speed: clip diff to ±MAX_JOINT_SPEED
                diff_limited = np.clip(diff, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)

                # Apply limited movement
                next_pos = current_pos + diff_limited
                action_dict[f"{name}.pos"] = next_pos

            # Debug: Print action dict structure on first step
            if step == 0:
                print("\n=== DEBUG: Action Dict Structure ===")
                for key, value in action_dict.items():
                    if key.endswith('.pos'):
                        print(f"  {key}: value={value}, type={type(value)}, dtype={value.dtype if hasattr(value, 'dtype') else 'N/A'}")
                print("=" * 80)

            # Send action to robot
            robot.send_action(action_dict)

            # Debug: Print actual sent actions every second
            if (step + 1) % 25 == 0:
                print("\n  Sent action dict to robot:")
                for name in joint_names[:3]:  # Just first 3 for brevity
                    print(f"    {name:15s}: {action_dict[f'{name}.pos']:8.4f} rad ({np.degrees(action_dict[f'{name}.pos']):7.2f}°)")

            # Circle detection (for policy input only, not for termination)
            # Use resized images for circle detection (same size as background model)
            left_circle = circle_detector.detect(obs_wrapped[left_key], "left")
            right_circle = circle_detector.detect(obs_wrapped[right_key], "right")

            # Display data (reduce frequency to avoid lag)
            if DISPLAY_DATA and step % 5 == 0:  # Log every 15 steps (~0.6s at 25 FPS)
                log_rerun_data(
                    observation=obs_wrapped,
                    action=action_dict,
                )

            # Print progress
            if (step + 1) % 25 == 0:  # Every second
                elapsed = time.time() - start_time
                fps_current = (step + 1) / elapsed
                # Show current, target, next actions and speed
                curr_pan = obs_wrapped['shoulder_pan.pos']
                next_pan = action_dict['shoulder_pan.pos']
                step_size = next_pan - curr_pan
                print(f"[{step+1:4d}/{max_steps}] FPS: {fps_current:.1f}, L={left_circle}, R={right_circle}")
                print(f"  current=[{obs_wrapped['shoulder_pan.pos']:.3f}, {obs_wrapped['shoulder_lift.pos']:.3f}, {obs_wrapped['elbow_flex.pos']:.3f}, ...]")
                print(f"  target =[{target_action[0]:.3f}, {target_action[1]:.3f}, {target_action[2]:.3f}, ...]")
                print(f"  next   =[{action_dict['shoulder_pan.pos']:.3f}, {action_dict['shoulder_lift.pos']:.3f}, {action_dict['elbow_flex.pos']:.3f}, ...] (step={step_size:.4f} rad)")

            step += 1

            # Maintain FPS
            step_duration = time.time() - step_start
            sleep_time = (1.0 / FPS) - step_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Episode complete
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

    if dataset and SAVE_DATASET:
        log_say("Pushing dataset to hub...")
        dataset.push_to_hub()

    log_say("Done!")


if __name__ == "__main__":
    main()
