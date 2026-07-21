import numpy as np
import time

# For controlling the Franka:
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.log_utils import get_deoxys_example_logger

# Robotiq gripper (pyrobotiqgripper) is imported lazily inside __init__ only
# when gripper_mode == "robotiq", so a Franka-only machine without the library
# can still run in gripper_mode == "franka".

from robo_utils.conversion_utils import transformation_to_pose

logger = get_deoxys_example_logger()

OPEN = 1
CLOSED = -1

# Robotiq gripper byte sent to deoxys is unused (gripper driven separately).
GRIPPER_NOOP = 0.0


class FrankaPandaController:

    def __init__(self, gripper_mode: str = "robotiq"):

        assert gripper_mode in ("robotiq", "franka"), \
            f"gripper_mode must be 'robotiq' or 'franka', got '{gripper_mode}'"
        self.gripper_mode = gripper_mode

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

        # Franka gripper action bytes (deoxys action vector last element).
        # Used to drive the Franka hand in gripper_mode == "franka"; inert in
        # gripper_mode == "robotiq" (kept for caller API compatibility).
        self.open_gripper_action = 1.0    # OPEN
        self.close_gripper_action = 0.0   # CLOSED

        if self.gripper_mode == "robotiq":
            # Robotiq gripper setup (separate USB/serial device).
            import pyrobotiqgripper as rq  # lazy: only needed for robotiq mode
            self.gripper = rq.RobotiqGripper()
            self.gripper.activate()
            self.gripper.calibrate(closemm=0, openmm=40)
            self.gripper.open()
            # Threshold (mm) used to classify Robotiq state as OPEN vs CLOSED.
            self._gripper_open_threshold_mm = 20.0
        else:
            # Franka hand: no external device. Held byte streamed into every
            # move action; updated only by open_gripper()/close_gripper().
            self.gripper = None
            self._gripper_byte = self.open_gripper_action  # start open

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
            # In franka mode get_gripper_state reads _gripper_state_buffer, so
            # wait for it too. In robotiq mode gripper state is external.
            gripper_ready = (
                self.gripper_mode != "franka"
                or len(self.robot_interface._gripper_state_buffer) > 0
            )
            if len(self.robot_interface._state_buffer) > 0 and gripper_ready:
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
        if self.gripper_mode == "robotiq":
            position_mm = self.gripper.position_mm()
            return OPEN if position_mm >= self._gripper_open_threshold_mm else CLOSED
        # franka: read Franka hand width from deoxys gripper state buffer.
        while True:
            if len(self.robot_interface._gripper_state_buffer) > 0:
                gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                # 0. is open and 1. is closed for gripper width
                return OPEN if np.abs(gripper_width) < 0.01 else CLOSED
            print("Waiting for franka gripper state...")

    def open_gripper(self, num_steps: int = 10):
        if self.gripper_mode == "robotiq":
            # num_steps kept for signature compatibility; Robotiq blocks until done.
            del num_steps
            self.gripper.open()
            return
        # franka: hold open byte and stream it so the Franka hand actuates.
        self._gripper_byte = self.open_gripper_action
        self._stream_gripper_byte(num_steps)

    def close_gripper(self, num_steps: int = 10):
        if self.gripper_mode == "robotiq":
            # num_steps kept for signature compatibility; Robotiq blocks until done.
            del num_steps
            self.gripper.close()
            return
        # franka: hold close byte and stream it so the Franka hand actuates.
        self._gripper_byte = self.close_gripper_action
        self._stream_gripper_byte(num_steps)

    def _stream_gripper_byte(self, num_steps: int):
        """Franka-only: stream current joints + held gripper byte for num_steps
        control ticks so the Franka hand actuates while the arm holds still."""
        current_joints = self.get_robot_joints()
        for _ in range(num_steps):
            action = np.concatenate([current_joints, [self._gripper_byte]]).tolist()
            self.robot_interface.control(
                controller_type=self.joint_controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )

    def _move_byte(self) -> float:
        """Gripper byte to fold into every move action: inert for robotiq,
        the held open/close byte for franka."""
        return GRIPPER_NOOP if self.gripper_mode == "robotiq" else self._gripper_byte

    def move_to_joints(
        self,
        target_joints: np.ndarray,
        gripper_state: int,
        max_iterations: int = 100,
        joint_error_threshold: float = 2e-3,
    ):

        assert type(target_joints) == np.ndarray, "Target joints must be a numpy array"
        assert target_joints.shape == (7,), "Target joints must be a 7D array"

        # gripper_state arg ignored in both modes. Robotiq is driven via
        # open_gripper/close_gripper (byte inert). Franka streams the held byte
        # set by open_gripper/close_gripper. See design doc for the safety
        # rationale (avoids dropping the object mid-motion).
        del gripper_state

        for _ in range(max_iterations):

            current_joints = self.get_robot_joints()

            joint_error = np.max(np.abs(current_joints - target_joints))
            if joint_error < joint_error_threshold:
                break

            action = np.concatenate([target_joints, [self._move_byte()]])
            action = action.tolist()

            self.robot_interface.control(
                controller_type=self.joint_controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )

    def move_along_trajectory(
        self,
        trajectory: np.ndarray,
        gripper_state: int,
        downsample_factor: int = 10,
        ticks_per_waypoint: int = 50,
        final_threshold: float = 2e-3,
        final_max_iterations: int = 200,
    ):
        """
        Stream the robot along a joint trajectory smoothly.

        cuRobo emits dense trajectories (interpolation_dt=0.01 -> ~100Hz).
        deoxys control() ticks at ~1kHz with SMOOTH_JOINT_POSITION interpolator.
        Sending every dense setpoint causes the interpolator to chase
        constantly-shifting targets -> jitter. Instead we:
          1. Downsample to give the interpolator headroom between setpoints.
          2. Hold each intermediate setpoint for a fixed number of ticks so
             the interpolator can complete a smoothing window.
          3. Settle on the final waypoint to a tight tolerance.

        Args:
            trajectory: (N, 7) array of joint positions for each waypoint.
            gripper_state: Retained for API compatibility; Robotiq gripper is
                driven separately via open_gripper/close_gripper.
            downsample_factor: Keep every Nth waypoint (last waypoint always kept).
            ticks_per_waypoint: Number of control() calls per intermediate waypoint
                (~ticks_per_waypoint ms at 1kHz FCI rate).
            final_threshold: Joint-error tolerance (rad) for the last waypoint.
            final_max_iterations: Max control() calls for the last waypoint.
        """
        assert type(trajectory) == np.ndarray, "Trajectory must be a numpy array"
        assert trajectory.ndim == 2 and trajectory.shape[1] == 7, "Trajectory must be an (N, 7) array"
        assert downsample_factor >= 1, "downsample_factor must be >= 1"

        del gripper_state  # ignored; gripper held via open_gripper/close_gripper.

        # Downsample, always keeping the final waypoint.
        if downsample_factor > 1 and len(trajectory) > 1:
            kept = trajectory[::downsample_factor]
            if not np.allclose(kept[-1], trajectory[-1]):
                kept = np.vstack([kept, trajectory[-1:]])
            sparse = kept
        else:
            sparse = trajectory

        last_idx = len(sparse) - 1
        for i, target_joints in enumerate(sparse):
            action = np.concatenate([target_joints, [self._move_byte()]]).tolist()

            if i == last_idx:
                # Settle to tight tolerance on final waypoint.
                for _ in range(final_max_iterations):
                    current_joints = self.get_robot_joints()
                    if np.max(np.abs(current_joints - target_joints)) < final_threshold:
                        break
                    self.robot_interface.control(
                        controller_type=self.joint_controller_type,
                        action=action,
                        controller_cfg=self.joint_controller_cfg,
                    )
            else:
                # Hold setpoint for a fixed window so interpolator can blend.
                for _ in range(ticks_per_waypoint):
                    self.robot_interface.control(
                        controller_type=self.joint_controller_type,
                        action=action,
                        controller_cfg=self.joint_controller_cfg,
                    )

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
            # Gripper byte: inert for robotiq, held byte for franka.
            action = action_pos.tolist() + action_axis_angle.tolist() + [self._move_byte()]
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
    controller.move_to_joints(controller.home_joints, gripper_state=OPEN)
