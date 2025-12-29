import glob
import os
import numpy as np

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from gello.env import RobotEnv
from gello.robots.dynamixel import DynamixelRobot
from gello.zmq_core.robot_node import ZMQClientRobot

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

class Gello:

    def __init__(self, args):
        self.robot_client = ZMQClientRobot(
            port=args.robot_port, host=args.hostname
        )
        self.env = RobotEnv(self.robot_client, control_rate_hz=args.control_rate_hz)
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

        self.robot = self.dynamixel_config.make_robot(self.port)

        # # For gripper linear mapping:
        # sim_bound = np.sum(
        #     self.sim.gripper_upper_limits - self.sim.gripper_lower_limits
        # ) - np.sum(self.sim.gripper_lower_limits)
        # gello_bound = args["gripper_close"] - args["gripper_open"]
        # self.grip_ratio = sim_bound / gello_bound  # m = (y2-y1/x2-x1)
        # self.grip_offset = (
        #     np.sum(self.sim.gripper_lower_limits)
        #     - self.grip_ratio * args["gripper_open"]
        # )  # c = y - mx

        # self.pressed = False
        # self.pcd_data = []
        # self.seg_data = []
        # self.pcd_lens = []

        # # Prepare data folder:
        # self.folder_path = args["demo_folder"] + args["exp_name"] + "/"
        # os.makedirs(self.folder_path, exist_ok=True)

        # self.num_demos = len(os.listdir(self.folder_path))
        # self.demo_folder = self.folder_path + "demo_" + str(self.num_demos) + "/"
        # os.makedirs(self.demo_folder, exist_ok=True)

        # self.steps = 0

    def get_usb_port(self):
        usb_ports = glob.glob("/dev/serial/by-id/*")
        print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            port = usb_ports[0]
            print(f"using port {port}")
        else:
            raise ValueError("No gello port found, please specify one or plug in gello")

        return port
    
    def collect(self):
        self.env.step(np.zeros((8,)))

        _ = input("Press Enter after holding Gello at home position")

        view_home = self.env.get_obs()["joint_positions"]
        gello_home = np.array(self.robot.get_joint_state())

        coeffs = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        while True:
            gello_joints = np.array(self.robot.get_joint_state())
            view_actuation = coeffs * (gello_joints - gello_home) + view_home

            # In gello, when the gripper is pressed, it returns 1., but franka in pybullet for the same state is approximately [1e-6, 1e-6]
            # Conversely, in open position the gello gripper is at 0.017, but franka in pybullet for the same state is approximately [0.039, 0.039]
            # We can use a linear function to map these two

            # sim_actuation = (
            #     coeffs[:-1] * (gello_joints[:-1] - gello_home[:-1]) + sim_home
            # )
            # self.control_grasp(gello_joints[-1])
            # self.sim.go_to_position(sim_actuation)
            # self.sim.actuate_gripper(sim_gripper_act)

            # for _ in range(10):
            #     self.sim.client_id.stepSimulation()  # Working with pybullet!
            self.env.step(view_actuation)
            # self.sim.control_view()


if __name__ == "__main__":

    args = RobotArgs()
    gello = Gello(args)
    gello.collect()