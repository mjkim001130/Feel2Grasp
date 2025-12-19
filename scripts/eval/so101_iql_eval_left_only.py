#!/usr/bin/env python
"""
SO101 Robot Evaluation with IQL Policy (Left Camera Only)

This version uses only the left camera and duplicates it for right camera input.
Right camera is broken, so we work around by using left camera for both inputs.

Usage:
    python so101_iql_eval_left_only.py
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
from iql_pretrained_policy_no_circle import IQLPreTrainedPolicyNoCircle
from circle_detector import CircleDetector


# ======================== Configuration ========================
# IQL Checkpoint paths
IQL_CHECKPOINT_PATH = "IQL_checkpoints/iql_step_855000_full_0_0.7_3.0.pt"
ENCODER_CHECKPOINT_PATH = "ae_out/encoder.pt"

# Episode configuration
NUM_EPISODES = 1
FPS = 25
EPISODE_TIME_SEC = 30
DISPLAY_DATA = True  # Disabled to improve FPS
DEVICE = "cuda"

# Action speed control
MAX_JOINT_SPEED = 12

# Circle detection
NUM_BG_FRAMES = 8
K_STD = 1

# Dataset configuration
SAVE_DATASET = False
DATASET_REPO_ID = "mjkim00/IQL_eval"


def main():
    init_logging()

    # ======================== Robot Configuration ========================
    # ONLY use front and left cameras (right camera is broken)
    # Left camera uses native resolution, will be resized immediately after get_observation()
    camera_config = {
        "front": OpenCVCameraConfig(index_or_path="/dev/camera_front", width=640, height=480, fps=FPS),
        "left": OpenCVCameraConfig(index_or_path="/dev/camera_left", width=3280, height=2464, fps=FPS),
        # RIGHT CAMERA DISABLED - broken
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

    # PATCH: Increase camera timeout for slow left camera (3280x2464)
    # Override get_observation to use 1000ms timeout instead of default 200ms
    def patched_get_observation():
        from lerobot.utils.errors import DeviceNotConnectedError

        if not robot.is_connected:
            raise DeviceNotConnectedError(f"{robot} is not connected.")

        # Read state from motors
        obs_dict = robot.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # Read images with INCREASED timeout (1000ms instead of 200ms)
        for cam_key, cam in robot.cameras.items():
            obs_dict[cam_key] = cam.async_read(timeout_ms=1000)  # PATCHED: 200 -> 1000

        return obs_dict

    robot.get_observation = patched_get_observation
    log_say("Camera timeout increased to 1000ms for left camera")

    # ======================== Initialize Circle Detector ========================
    log_say("Initializing circle detector (LEFT CAMERA ONLY)...")
    log_say(f"Collecting {NUM_BG_FRAMES} background frames - keep robot stationary!")

    circle_detector = CircleDetector(num_bg_frames=NUM_BG_FRAMES, k_std=K_STD, use_center_crop=True, history_size=10)

    # Collect background frames
    obs = robot.get_observation()
    print(f"Available observation keys: {list(obs.keys())}")

    # Find left camera key
    left_key = None
    for key in obs.keys():
        key_lower = key.lower()
        if 'left' in key_lower and not key_lower.endswith('.pos'):
            left_key = key

    if left_key is None:
        log_say("Warning: Could not find left camera key. Circle detection disabled.")
        print(f"Available keys: {list(obs.keys())}")
        circle_detector = None
    else:
        print(f"Using left camera: {left_key}")
        print("NOTE: Right camera is broken, will duplicate left camera for right input")

        import cv2
        for i in range(NUM_BG_FRAMES):
            obs = robot.get_observation()

            # Fast downsample using slicing instead of resize (much faster than cv2.resize)
            # 3280x2464 -> 640x480: take every 5th pixel in width, every 5.13th in height
            # Approximate: [::5, ::5] gives roughly 656x493, then resize to exact 640x480
            left_img_downsampled = obs[left_key][::5, ::5]  # Fast slicing
            obs[left_key] = cv2.resize(left_img_downsampled, (640, 480))  # Small resize is fast

            # Use same left image for both left and right background
            circle_detector.add_background_frame(obs[left_key], "left")
            circle_detector.add_background_frame(obs[left_key], "right")

            time.sleep(1.0 / FPS)

            if (i + 1) % 10 == 0:
                print(f"  Collected {i + 1}/{NUM_BG_FRAMES} frames")

        log_say("Circle detector initialized (using left camera for both inputs)!")
        print(f"Status: {circle_detector.get_status()}")

    # ======================== Load IQL Policy ========================
    log_say("Loading IQL policy...")
    # policy = IQLPreTrainedPolicy(
    #     checkpoint_path=IQL_CHECKPOINT_PATH,
    #     encoder_path=ENCODER_CHECKPOINT_PATH,
    #     circle_detector=circle_detector,
    #     device=DEVICE,
    # )
    policy = IQLPreTrainedPolicyNoCircle(
        checkpoint_path=IQL_CHECKPOINT_PATH,
        encoder_path=ENCODER_CHECKPOINT_PATH,
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

        # Adjust image shapes
        dataset_features['observation.images.left']['shape'] = (480, 640, 3)

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
        init_rerun(session_name="iql_evaluation_left_only")

    # ======================== Initialize Keyboard Listener ========================
    listener, events = init_keyboard_listener()

    # ======================== Run Episode ========================
    log_say(f"Starting IQL evaluation episode (max {EPISODE_TIME_SEC}s)")
    log_say(f"Using speed limit: {MAX_JOINT_SPEED} rad/step = {MAX_JOINT_SPEED * FPS:.2f} rad/s = {np.degrees(MAX_JOINT_SPEED * FPS):.1f}°/s")
    log_say("Press 'q' to stop at any time")
    log_say("Circle detection uses LEFT camera only (right camera broken)")

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
        print(f"  Left image shape (original): {obs_init[left_key].shape}")
        print(f"  Left image shape (resized for policy): {left_resized.shape}")
        print(f"  NOTE: Right camera BROKEN - using left camera for both inputs")
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

            # Fast downsample + resize (matching background collection method)
            import cv2
            if left_key and left_key in obs:
                original_shape = obs[left_key].shape if step == 0 else None
                # Fast slicing downsample then small resize (much faster than direct cv2.resize)
                left_img_downsampled = obs[left_key][::5, ::5]
                obs[left_key] = cv2.resize(left_img_downsampled, (640, 480))
                # DUPLICATE left camera for right camera (since right is broken)
                obs['right'] = obs[left_key].copy()

                # Debug: Print image shapes on first step
                if step == 0:
                    print(f"\n[DEBUG] Image Resize (matching record_loop):")
                    print(f"  Original left image shape: {original_shape}")
                    print(f"  Resized left image shape: {obs[left_key].shape}")
                    print(f"  Duplicated right image shape: {obs['right'].shape}")

            # DO NOT wrap joint angles - policy was trained with unwrapped angles
            obs_wrapped = obs

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

                # Clip gripper to training data range (0-42)
                # Gripper uses MotorNormMode.RANGE_0_100 (percentage, not radians)
                if name == "gripper":
                    next_pos = np.clip(next_pos, 0.0, 40.0)

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

            # Skip debug readback for performance (it adds 0.1s delay per step)

            # Debug: Print actual sent actions every second
            if (step + 1) % 25 == 0:
                print("\n  Sent action dict to robot:")
                for name in joint_names[:3]:  # Just first 3 for brevity
                    print(f"    {name:15s}: {action_dict[f'{name}.pos']:8.4f} rad ({np.degrees(action_dict[f'{name}.pos']):7.2f}°)")

            # Circle detection (use left camera for both)
            if circle_detector is not None:
                left_circle = circle_detector.detect(obs_wrapped[left_key], "left")
                # Use same left image for right detection
                right_circle = circle_detector.detect(obs_wrapped[left_key], "right")
            else:
                left_circle = 0
                right_circle = 0

            # Display data (reduce frequency to avoid lag)
            if DISPLAY_DATA and step % 1 == 0:  # Log every 5 steps
                log_rerun_data(
                    observation=obs_wrapped,
                    action=action_dict,
                )

            # if DISPLAY_DATA and step % 1 == 0:
            #     display_obs = {
            #         'front': obs_wrapped['front'],
            #         left_key: obs_wrapped[left_key],
            #         #right_key: obs_wrapped[right_key],
            #     }
            #     log_rerun_data(observation=display_obs, action=action_dict)


            # Print progress
            if (step + 1) % 25 == 0:  # Every second
                elapsed = time.time() - start_time
                fps_current = (step + 1) / elapsed
                curr_pan = obs_wrapped['shoulder_pan.pos']
                next_pan = action_dict['shoulder_pan.pos']
                step_size = next_pan - curr_pan
                print(f"[{step+1:4d}/{max_steps}] FPS: {fps_current:.1f}, L={left_circle}, R={right_circle} (R=duplicate)")
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
