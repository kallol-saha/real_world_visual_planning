"""
Standalone move-to-home + grasp smoke test (raw deoxys, no controller class).

Move to home joints, then close the gripper.

    python actuate_robot.py                # robotiq (default)
    python actuate_robot.py --mode franka  # franka hand (deoxys action byte)

Run from repo root (configs/ paths are relative). Keep e-stop in reach.
"""

import argparse
import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

# Franka gripper action bytes (deoxys action vector last element).
OPEN_BYTE = 1.0
CLOSE_BYTE = 0.0

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["robotiq", "franka"], default="robotiq")
parser.add_argument("--gripper_steps", type=int, default=50,
                    help="franka mode: control ticks streamed to actuate the hand.")
args = parser.parse_args()

config_path = "configs"

robot_interface = FrankaInterface(
    f"{config_path}/charmander.yml",
    use_visualizer=False
)

controller_type = "JOINT_POSITION"
controller_cfg = YamlConfig(
    f"{config_path}/joint-position-controller.yml"
).as_easydict()

# Gripper setup: robotiq uses a separate device; franka is driven via the byte.
gripper = None
if args.mode == "robotiq":
    import pyrobotiqgripper as rq  # lazy: only needed for robotiq mode
    gripper = rq.RobotiqGripper()
    gripper.activate()
    gripper.calibrate(closemm=0, openmm=40)
    gripper.open()

# These are home joints:
target_joint_positions = [
    -1.3159,
    -0.4246,
     0.1067,
    -2.7110,
    -0.0562,
     2.3219,
     0.7518,
]

# Byte folded into the move action: inert (unused) for robotiq; open for franka
# so the hand stays open while moving to home.
move_byte = CLOSE_BYTE if args.mode == "robotiq" else OPEN_BYTE
action = target_joint_positions + [move_byte]

while True:

    if len(robot_interface._state_buffer) > 0:

        if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_joint_positions))) < 2e-3:
            break

    robot_interface.control(
        controller_type=controller_type,
        action=action,
        controller_cfg=controller_cfg,
    )

# Close the gripper (grasp).
if args.mode == "robotiq":
    gripper.close()
else:
    # franka: stream home joints + close byte so the Franka hand actuates.
    close_action = target_joint_positions + [CLOSE_BYTE]
    for _ in range(args.gripper_steps):
        robot_interface.control(
            controller_type=controller_type,
            action=close_action,
            controller_cfg=controller_cfg,
        )

robot_interface.close()
