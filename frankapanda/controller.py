import numpy as np
import time

# For controlling the Franka:
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.log_utils import get_deoxys_example_logger

from robo_utils.conversion_utils import transformation_to_pose

logger = get_deoxys_example_logger()

OPEN = 1
CLOSED = -1

class FrankaPandaController:

    def __init__(self):

        self.robot_interface = FrankaInterface(
            "configs/charmander.yml", 
            use_visualizer=False
        )

        self.joint_controller_cfg = YamlConfig(
            "configs/joint-position-controller.yml"
        ).as_easydict()
        self.joint_controller_type = "JOINT_POSITION"

        self.osc_controller_cfg = YamlConfig(
            "configs/tuned-osc-yaw-controller.yml"
        ).as_easydict()
        self.osc_controller_type = "OSC_POSE"

        # Changed to this for shelf packing
        self.home_joints = np.array([
            -1.3159,
            -0.4246,
             0.1067,
            -2.7110,
            -0.0562,
             2.3219,
             0.7518,
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
    
    def check_joint_position_violation(self):

        violation = self.robot_interface._state_buffer[-1].last_motion_errors.joint_position_limits_violation
        return violation
    
    def get_robot_joints(self) -> np.ndarray:

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                robot_joints = self.robot_interface._state_buffer[-1].q
                if self.check_joint_position_violation():
                    print("Joint position violation detected!")
                return np.array(robot_joints)
            print("Waiting for robot joints...")
    
    def get_qpos(self) -> np.ndarray:
        while True:
            if len(self.robot_interface._state_buffer) > 0 and len(self.robot_interface._gripper_state_buffer) > 0:
                joint_positions = self.robot_interface._state_buffer[-1].q
                gripper_state = self.get_gripper_state()
                qpos = np.concatenate([joint_positions, [gripper_state]])
                if self.check_joint_position_violation():
                    print("Joint position violation detected!")
                return qpos
            print("Waiting for robot qpos...")
    
    def get_gripper_pose(self, as_transform=False, format='wxyz') -> np.ndarray:
        while True:
            if len(self.robot_interface._state_buffer) > 0:
                gripper_pose = self.robot_interface._state_buffer[-1].O_T_EE
                gripper_pose = np.array(gripper_pose).reshape(4, 4).T
                if self.check_joint_position_violation():
                    print("Joint position violation detected!")
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

            if gripper_state == OPEN:
                gripper_state = self.close_gripper_action
            elif gripper_state == CLOSED:
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

    def move_along_trajectory(self, trajectory: np.ndarray, max_iterations_per_waypoint: int = 100):
        """
        Move the robot along a joint trajectory.

        Args:
            trajectory: (N, 7) array of joint positions for each waypoint
            max_iterations_per_waypoint: Maximum control iterations per waypoint
        """
        assert type(trajectory) == np.ndarray, "Trajectory must be a numpy array"
        assert trajectory.ndim == 2 and trajectory.shape[1] == 7, "Trajectory must be an (N, 7) array"

        gripper_state = self.get_gripper_state()
        if gripper_state == OPEN:
            gripper_action = self.close_gripper_action
        elif gripper_state == CLOSED:
            gripper_action = self.open_gripper_action

        final_joints = trajectory[-1]

        for waypoint_idx, target_joints in enumerate(trajectory):
            current_joints = self.get_robot_joints()

            action = np.concatenate([target_joints, [gripper_action]])
            action = action.tolist()

            self.robot_interface.control(
                controller_type=self.joint_controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )

        self.move_to_joints(final_joints)

    def osc_move(self, target_pose, num_steps):
        """
        Move to a target pose using OSC controller.
        
        Args:
            target_pose: Tuple of (target_pos, target_quat) where
                target_pos is (3, 1) array and target_quat is (4,) array
            num_steps: Number of control steps to execute
        """
        target_pos, target_quat = target_pose
        
        # Wait for robot state
        while len(self.robot_interface._state_buffer) == 0:
            logger.warn("Robot state not received, waiting...")
            time.sleep(0.5)

        for _ in range(num_steps):
            # Get current pose
            current_pose = self.robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3:]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)
            
            # Ensure quaternions are in the same hemisphere
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            
            # Compute quaternion difference and convert to axis-angle
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            
            # Compute position and rotation actions
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            
            # Clip actions to safe limits
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            # Combine actions: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper]
            action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]
            logger.info(f"Axis angle action {action_axis_angle.tolist()}")
            
            self.robot_interface.control(
                controller_type=self.osc_controller_type,
                action=action,
                controller_cfg=self.osc_controller_cfg,
            )
        
        return action

    def move_to_target_pose(
        self,
        target_delta_pose,
        num_steps,
        num_additional_steps=0,
    ):
        """
        Move to a target pose specified as a delta from current pose.
        
        Args:
            target_delta_pose: Array of shape (6,) containing [delta_x, delta_y, delta_z, delta_rot_x, delta_rot_y, delta_rot_z]
                where rotation deltas are in axis-angle representation
            num_steps: Number of control steps for initial movement
            num_additional_steps: Number of additional control steps for fine-tuning (default: 0)
        """
        # Wait for robot state
        while len(self.robot_interface._state_buffer) == 0:
            logger.warn("Robot state not received, waiting...")
            time.sleep(0.5)

        # Parse target delta pose
        target_delta_pos = np.array(target_delta_pose[:3])
        target_delta_axis_angle = np.array(target_delta_pose[3:])

        # Get current pose
        current_ee_pose = self.robot_interface.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        current_rot = current_ee_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        # Compute target pose
        target_pos = target_delta_pos.reshape(3, 1) + current_pos
        target_axis_angle = target_delta_axis_angle + current_axis_angle
        target_quat = transform_utils.axisangle2quat(target_axis_angle)

        logger.info(f"Target axis angle: {target_axis_angle}")
        logger.info(f"Target quaternion: {target_quat}")

        # Move to target pose
        self.osc_move((target_pos, target_quat), num_steps)
        
        # Fine-tune with additional steps if specified
        if num_additional_steps > 0:
            self.osc_move((target_pos, target_quat), num_additional_steps)
    
if __name__ == "__main__":
    controller = FrankaPandaController()
    controller.get_gripper_pose()
    controller.move_to_joints(controller.home_joints)