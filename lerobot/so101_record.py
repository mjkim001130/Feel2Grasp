import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    make_robot_from_config,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    make_teleoperator_from_config,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_devices
#from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.scripts.lerobot_record import RecordConfig, record_loop, DatasetRecordConfig

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

# ======================== Dataset Record Config ========================
NUM_EPISODES = 1
FPS = 25
EPISODE_TIME_SEC = 40
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick the pen from the hole and up towards"
REPO_ID = "mjkim00/test"
SINGLE_TASK = True
DISPLAY_DATA = True
RESUME = False

rc_dataset = DatasetRecordConfig(REPO_ID, single_task=SINGLE_TASK)
rc_dataset.num_episodes = NUM_EPISODES
rc_dataset.fps = FPS
rc_dataset.episode_time_s = EPISODE_TIME_SEC
rc_dataset.reset_time_s = RESET_TIME_SEC

# ======================== ============= ========================
# ======================== Record Config ========================
# ======================== ============= ========================

# Create the robot and teleoperator configurations
# camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=25),
#                  }

# front_cam = "/dev/camera_front"
# left_cam = "/dev/camera_left"
# right_cam = "/dev/camera_right"

camera_config = {"front": OpenCVCameraConfig(index_or_path="/dev/camera_front", width=640, height=480, fps=25),
                 "left": OpenCVCameraConfig(index_or_path="/dev/camera_left", width=3280, height=2464, fps=25),
                 "right": OpenCVCameraConfig(index_or_path="/dev/camera_right", width=3280, height=2464, fps=25),
                 }

# "left": OpenCVCameraConfig(index_or_path=left_cam, width=3280, height=2464, fps=25),
# "right": OpenCVCameraConfig(index_or_path=right_cam, width=3280, height=2464, fps=25),
                 
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM1", id="black_follower_arm", cameras=camera_config
)
teleop_config = SO101LeaderConfig(port="/dev/ttyACM0", id="white_leader_arm")

# Initialize the robot and teleoperator
robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)

rc = RecordConfig(robot=robot,
                  teleop=teleop,
                  dataset=rc_dataset)
rc.display_data = DISPLAY_DATA
rc.resume = RESUME

# robot = make_robot_from_config(rc.robot)
# teleop = make_teleoperator_from_config(rc.teleop)
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

dataset_features = combine_feature_dicts(
        # action: 6-DoF Joints
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=rc.dataset.video,
        ),
        # observation: Eye-on-hand Image
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=rc.dataset.video,
        ),
)


dataset_features['observation.images.left']['shape'] = (480, 640, 3)
dataset_features['observation.images.right']['shape'] = (480, 640, 3)

# ======================== Lebot Dataset Settings ========================
if rc.resume:
    dataset = LeRobotDataset(
        rc.dataset.repo_id,
        root=rc.dataset.root,
        batch_encoding_size=rc.dataset.video_encoding_batch_size,
    )

    if hasattr(robot, "cameras") and len(robot.cameras) > 0:
        dataset.start_image_writer(
            num_processes=rc.dataset.num_image_writer_processes,
            num_threads=rc.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        )
    sanity_check_dataset_robot_compatibility(dataset, robot, rc.dataset.fps, dataset_features)
else:
    sanity_check_dataset_name(rc.dataset.repo_id, rc.policy)
    dataset = LeRobotDataset.create(
        rc.dataset.repo_id,
        rc.dataset.fps,
        root=rc.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=rc.dataset.video,
        image_writer_processes=rc.dataset.num_image_writer_processes,
        image_writer_threads=rc.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=rc.dataset.video_encoding_batch_size,
    )

# ======================== Policy Settings ========================
policy = None if rc.policy is None else make_policy(rc.policy, ds_meta=dataset.meta)
preprocessor = None
postprocessor = None
if rc.policy is not None:
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=rc.policy,
        pretrained_path=rc.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, rc.dataset.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": rc.policy.device},
            "rename_observations_processor": {"rename_map": rc.dataset.rename_map},
        },
    )

# ======================== Connect ======================
# Connect the robot and teleoperator
robot.connect()
teleop.connect()

init_logging()
if rc.display_data:
        init_rerun(session_name="recording")

listener, events = init_keyboard_listener()

with VideoEncodingManager(dataset):
    recorded_episodes = 0
    while recorded_episodes < rc.dataset.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {dataset.num_episodes}", rc.play_sounds)
        record_loop(
            robot=robot,
            events=events,
            fps=rc.dataset.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=rc.dataset.episode_time_s,
            single_task=rc.dataset.single_task,
            display_data=rc.display_data,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < rc.dataset.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", rc.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=rc.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=rc.dataset.reset_time_s,
                single_task=rc.dataset.single_task,
                display_data=rc.display_data,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode", rc.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    log_say("Stop recording", rc.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if rc.dataset.push_to_hub:
        dataset.push_to_hub(tags=rc.dataset.tags, private=rc.dataset.private)

    log_say("Exiting", rc.play_sounds)