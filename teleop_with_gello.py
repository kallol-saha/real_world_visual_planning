import glob
import os
import numpy as np

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# For Gello:
from gello.env import RobotEnv
from gello.robots.dynamixel import DynamixelRobot
from gello.zmq_core.robot_node import ZMQClientRobot

# For controlling the Franka:
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

# For camera:
from pyk4a import PyK4A, connected_device_count
from pyk4a.pyk4a import PyK4A as PyK4A_device
import cv2
import numpy as np
import time

@dataclass
class RobotArgs:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    control_rate_hz: int = 100
    gripper_close: float = 1.
    gripper_open: float = 0.01771439

@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of GELLO. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )

class GelloTeleop:

    def __init__(self, args):

        # -------------- FOR GELLO -------------- #

        self.gello_gripper_open = args.gripper_open
        self.gello_gripper_closed = args.gripper_close
        
        self.gello_client = ZMQClientRobot(
            port=args.robot_port, host=args.hostname
        )
        self.env = RobotEnv(self.gello_client, control_rate_hz=args.control_rate_hz)
        self.port = self.get_usb_port()

        self.dynamixel_config = DynamixelRobotConfig(
            joint_ids=(1, 2, 3, 4, 5, 6, 7),
            joint_offsets=(
                3 * np.pi / 2,
                2 * np.pi / 2,
                1 * np.pi / 2,
                4 * np.pi / 2,
                -2 * np.pi / 2 + 2 * np.pi,
                3 * np.pi / 2,
                4 * np.pi / 2,
            ),
            joint_signs=(1, -1, 1, 1, 1, -1, 1),
            gripper_config=(8, 195, 152),
        )

        self.gello = self.dynamixel_config.make_robot(self.port)

        # -------------------------------------- #

        # -------------- FOR FRANKA -------------- #

        self.robot_interface = FrankaInterface(
        "charmander.yml", 
        use_visualizer=False
    )

        self.controller_cfg = YamlConfig(
            "joint-position-controller.yml"
        ).as_easydict()
        self.controller_type = "JOINT_POSITION"

        self.home_joints = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]
        # -------------------------------------- #

    def get_usb_port(self):
        usb_ports = glob.glob("/dev/serial/by-id/*")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            port = usb_ports[0]
            print(f"using port {port}")
        else:
            raise ValueError("No gello port found, please specify one or plug in gello")

        return port
    
    def get_robot_joints(self):

        while True:

            if len(self.robot_interface._state_buffer) > 0:
                robot_joints = self.robot_interface._state_buffer[-1].q
                return robot_joints
    
    def get_gripper_pose(self):
        while True:
            if len(self.robot_interface._state_buffer) > 0:
                gripper_pose = self.robot_interface._state_buffer[-1].O_T_EE
                gripper_pose = np.array(gripper_pose).reshape(4, 4)
                return gripper_pose
    
    def get_gripper_state(self):

        while True:

            if len(self.robot_interface._gripper_state_buffer) > 0:
                gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                gripper_state = 0. if np.abs(gripper_width) < 0.01 else 1.
                return gripper_state
    
    def actuate_robot(self, robot_joints, gripper_action = 1.):

        action = list(robot_joints) + [gripper_action]
        max_iterations = 60
        iterations = 0

        while iterations < max_iterations:

            if len(self.robot_interface._gripper_state_buffer) > 0:

                gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                gripper_state = 0. if np.abs(gripper_width) < 0.01 else 1.

                joint_error = np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(robot_joints)))
                gripper_error = np.abs(gripper_state - gripper_action)

                if joint_error < 1e-3 and gripper_error < 1e-3:
                    break

            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
            iterations += 1
    
    
    
    
    def control(self):
        self.env.step(np.zeros((8,)))
        # self.actuate_robot(self.home_joints)

        # Initialize the camera
        k4a = PyK4A(device_id=1)
        k4a.start()

        _ = input("Press Enter after holding Gello at home position")

        view_home = self.env.get_obs()["joint_positions"]
        gello_home = np.array(self.gello.get_joint_state())
        robot_home = self.get_robot_joints()

        coeffs = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.8\
        
        joint_data = np.array([], dtype=np.float32).reshape(0, 7)  # 7 joints
        gripper_data = np.array([], dtype=np.float32)  # 1D gripper state
        gripper_pose_data = np.array([], dtype=np.float32).reshape(0, 4, 4)  # 4x4 gripper pose
        rgb_data = np.array([], dtype=np.uint8).reshape(0, 720, 1280, 3)  # RGB images
        depth_data = np.array([], dtype=np.float32).reshape(0, 720, 1280)  # Depth images
        ir_data = np.array([], dtype=np.float32).reshape(0, 576, 640)  # IR images

        while True:
            try:
                gello_joints = np.array(self.gello.get_joint_state())
                view_actuation = coeffs * (gello_joints - gello_home) + view_home

                if len(self.robot_interface._state_buffer) > 0:
                    robot_actuation = (
                        coeffs[:-1] * (gello_joints[:-1] - gello_home[:-1]) + robot_home
                    )

                    # When gello gripper open --> 0.015
                    # When gello gripper closed --> 1.0
                    # When franka gripper open --> -0.071
                    # When franka gripper closed --> 0.

                    # robot_actuation[7:] = robot_home[7:]  # Debugging joints one by one

                    if gello_joints[-1] > 0.5:
                        gripper_action = 0.0
                    else:
                        gripper_action = 1.0
                    # print(gello_joints[-1], gripper_action)

                ir_frame, rgb_frame, ir_frame_norm, pcd_frame, depth_frame = get_kinect_rgbd_frame(k4a)
                cur_robot_joints = self.get_robot_joints()
                cur_gripper_state = self.get_gripper_state()
                cur_gripper_pose = self.get_gripper_pose()

                # Normalize the depth frame according to the max depth
                depth_frame = np.clip((depth_frame / 2100)* 255, 0, 255)
                depth_frame = depth_frame.astype(np.uint8)

                # Normalize the ir frame according to the max ir
                ir_frame = np.clip((ir_frame / 3000) * 255, 0, 255)
                ir_frame = ir_frame.astype(np.uint8)

                rgb_data = np.append(rgb_data, [rgb_frame[..., :3]], axis=0)
                depth_data = np.append(depth_data, [depth_frame], axis=0)
                ir_data = np.append(ir_data, [ir_frame], axis=0)
                joint_data = np.append(joint_data, [cur_robot_joints], axis=0)
                gripper_data = np.append(gripper_data, [cur_gripper_state], axis=0)
                gripper_pose_data = np.append(gripper_pose_data, [cur_gripper_pose], axis=0)

                # Keep saving every iteration:
                # np.savez("data5.npz", 
                #          rgb_data=rgb_data, 
                #          depth_data=depth_data, 
                #          ir_data=ir_data, 
                #          joint_data=joint_data, 
                #          gripper_data=gripper_data)
                
                # # If the gripper is closed, we need to open it
                # if cur_gripper_state == -1.0:
                #     gripper_action = 0.0

                # If the gripper is open, we need to close it
                self.env.step(view_actuation)
                self.actuate_robot(robot_actuation, gripper_action)
                
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected! Saving data and exiting...")
                # Save final data
                np.savez("data9.npz", 
                         rgb_data=rgb_data, 
                         depth_data=depth_data, 
                         ir_data=ir_data, 
                         joint_data=joint_data, 
                         gripper_data=gripper_data)
                print("Data saved to data5_final.npz")
                break

def get_kinect_rgbd_frame(device: PyK4A_device, visualize=False):
    """
    Capture an IR frame from the Kinect camera.
    """
    # Capture an IR frame
    rgb_frame = None
    capture = None
    for i in range(20):
        try:
            device.get_capture()
            capture = device.get_capture()
            if capture is not None:
                ir_frame = capture.ir

                # depth_frame = capture.depth
                # ---
                depth_frame = capture.transformed_depth
                # ---
                
                # cv2.imshow('IR', ir_frame)
                rgb_frame = capture.color
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                gray_frame = np.clip(gray_frame, 0, 5e3) / 5e3  # Clip and normalize
                # cv2.imshow('color', rgb_frame)
                
                ir_frame_norm = np.clip(ir_frame, 0, 5e3) / 5e3  # Clip and normalize
                pcd_frame = capture.transformed_depth_point_cloud
                # print(pcd_frame.shape, ir_frame.shape)
                # print("successful capture")
                return ir_frame, rgb_frame, ir_frame_norm, pcd_frame, depth_frame
        except:
            time.sleep(0.1)
            # print("Failed to capture IR frame.")
    else:
        # print("Failed to capture IR frame after 20 attempts.")
        return None




if __name__ == "__main__":

    args = RobotArgs()
    gello = GelloTeleop(args)
    gello.control()