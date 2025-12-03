from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower

camera_config = {
    "front": OpenCVCameraConfig(index_or_path="/dev/camera_front", width=640, height=480, fps=25),
    "left": OpenCVCameraConfig(index_or_path="/dev/camera_left", width=3280, height=2464, fps=25),
    "right": OpenCVCameraConfig(index_or_path="/dev/camera_right", width=3280, height=2464, fps=25),
}

robot_config = KochFollowerConfig(
    port="/dev/ttyACM1",
    id="black_follower_arm",
    cameras=camera_config
)

teleop_config = KochLeaderConfig(
    port="/dev/ttyACM0",
    id="white_leader_arm",
)

robot = KochFollower(robot_config)
teleop_device = KochLeader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)