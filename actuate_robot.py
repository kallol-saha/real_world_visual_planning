import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

robot_interface = FrankaInterface(
        "charmander.yml", 
        use_visualizer=False
    )

controller_cfg = YamlConfig(
    "joint-position-controller.yml"
).as_easydict()
controller_type = "JOINT_POSITION"

target_joint_positions = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]

action = target_joint_positions + [-1.0]    # Adding the gripper action

while True:

    if len(robot_interface._state_buffer) > 0:

        if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_joint_positions))) < 1e-3:
            break

    robot_interface.control(
        controller_type=controller_type,
        action=action,
        controller_cfg=controller_cfg,
    )

robot_interface.close()