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

from robo_utils.conversion_utils import transformation_to_pose

OPEN = 1
CLOSED = -1

class FrankaPandaController:

    def __init__(self):

        self.robot_interface = FrankaInterface(
            "charmander.yml", 
            use_visualizer=False
        )

        self.joint_controller_cfg = YamlConfig(
            "configs/joint-position-controller.yml"
        ).as_easydict()
        self.joint_controller_type = "JOINT_POSITION"

        # Changed to this for shelf packing
        self.home_joints = np.array([
            -0.83088868,
            0.08609554,
            0.09999037,
            -2.22055988,
            -0.05393224,
            2.32607312,
            0.35714266,
        ])
            # 0.09162008114028396,
            # -0.19826458111314524,
            # -0.01990020486871322,
            # -2.4732269941140346,
            # -0.01307073642274261,
            # 2.30396583422025,
            # 0.8480939705504309,
        # ])

        self.open_gripper_action = 1.0    # This is OPEN
        self.close_gripper_action = 0.0    # This is CLOSED
    
    def get_robot_joints(self) -> np.ndarray:

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                robot_joints = self.robot_interface._state_buffer[-1].q
                return np.array(robot_joints)
            print("Waiting for robot joints...")
    
    def get_qpos(self) -> np.ndarray:
        while True:
            if len(self.robot_interface._state_buffer) > 0 and len(self.robot_interface._gripper_state_buffer) > 0:
                joint_positions = self.robot_interface._state_buffer[-1].q
                gripper_state = self.get_gripper_state()
                qpos = np.concatenate([joint_positions, [gripper_state]])
                return qpos
            print("Waiting for robot qpos...")
    
    def get_gripper_pose(self, as_transform=False, format='wxyz') -> np.ndarray:
        while True:
            if len(self.robot_interface._state_buffer) > 0:
                gripper_pose = self.robot_interface._state_buffer[-1].O_T_EE
                gripper_pose = np.array(gripper_pose).reshape(4, 4).T
                if not as_transform:
                    gripper_pose = transformation_to_pose(gripper_pose, format=format)
                return gripper_pose
            print("Waiting for robot gripper pose...")
            
    def get_gripper_state(self) -> int:

        while True:
            if len(self.robot_interface._gripper_state_buffer) > 0:
                gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                gripper_state = OPEN if np.abs(gripper_width) < 0.01 else CLOSED    # 0. is open and 1. is closed for gripper width
                return gripper_state
    
    def open_gripper(self, num_steps: int = 10):

        current_joints = self.get_robot_joints()
        for _ in range(num_steps):
            action = np.concatenate([current_joints, [self.open_gripper_action]])
            self.robot_interface.control(
                controller_type=self.joint_controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )

    def close_gripper(self, num_steps: int = 10):

        current_joints = self.get_robot_joints()
        for _ in range(num_steps):
            action = np.concatenate([current_joints, [self.close_gripper_action]])
            self.robot_interface.control(
                controller_type=self.joint_controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )
    
    def move_to_joints(self, target_joints: np.ndarray, max_iterations: int = 100):

        assert type(target_joints) == np.ndarray, "Target joints must be a numpy array"
        assert target_joints.shape == (7,), "Target joints must be a 7D array"
        
        gripper_state = self.get_gripper_state()

        for _ in range(max_iterations):

            current_joints = self.get_robot_joints()

            if gripper_state == CLOSED:
                gripper_state = self.close_gripper_action
            elif gripper_state == OPEN:
                gripper_state = self.open_gripper_action

            joint_error = np.max(np.abs(current_joints - target_joints))
            if joint_error < 1e-3:
                break

            action = np.concatenate([target_joints, [gripper_state]])
            action = action.tolist()

            self.robot_interface.control(
                controller_type=self.joint_controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )
    
if __name__ == "__main__":
    controller = FrankaPandaController()
    controller.get_gripper_pose()
    controller.move_to_joints(controller.home_joints)