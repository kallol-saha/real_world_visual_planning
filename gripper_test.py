import numpy as np
import time
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

class GripperTest:
    def __init__(self):
        # Initialize Franka interface
        self.robot_interface = FrankaInterface(
            "charmander.yml", 
            use_visualizer=False
        )

        self.controller_cfg = YamlConfig(
            "joint-position-controller.yml"
        ).as_easydict()
        self.controller_type = "JOINT_POSITION"

        # Home position for the robot
        self.home_joints = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

    def get_gripper_state(self):
        """Get current gripper state (0 for open, -1 for closed)"""
        while True:
            if len(self.robot_interface._gripper_state_buffer) > 0:
                gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                gripper_state = 0.0 if np.abs(gripper_width) < 0.01 else -1.0
                return gripper_state

    def actuate_robot(self, robot_joints, gripper_action=-1.0):
        """Move robot to joint positions and set gripper state"""
        action = list(robot_joints) + [gripper_action]
        max_iterations = 60
        iterations = 0

        while iterations < max_iterations:
            if len(self.robot_interface._gripper_state_buffer) > 0:
                gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                gripper_state = 0.0 if np.abs(gripper_width) < 0.01 else -1.0

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

    def run_gripper_test(self, duration_seconds=60):
        """Run gripper open/close test for specified duration"""
        print("Starting gripper test...")
        print("Moving to home position first...")
        
        # Move to home position
        self.actuate_robot(self.home_joints, gripper_action=1.)# Start with open gripper
        print("Reached home position")
        
        start_time = time.time()
        cycle_count = 0
        
        while time.time() - start_time < duration_seconds:
            cycle_count += 1
            current_time = time.time() - start_time
            
            # Open gripper
            print(f"Cycle {cycle_count} ({current_time:.1f}s): Opening gripper...")
            self.actuate_robot(self.home_joints, gripper_action=0.0)
            current_gripper_state = self.get_gripper_state()
            print(f"Gripper state: {'OPEN' if current_gripper_state == 0.0 else 'CLOSED'}")
            
            # Wait 5 seconds
            time.sleep(5)
            
            # Close gripper
            print(f"Cycle {cycle_count} ({time.time() - start_time:.1f}s): Closing gripper...")
            self.actuate_robot(self.home_joints, gripper_action=-1.0)
            current_gripper_state = self.get_gripper_state()
            print(f"Gripper state: {'OPEN' if current_gripper_state == 0.0 else 'CLOSED'}")
            
            # Wait 5 seconds
            time.sleep(5)
        
        print(f"Test completed! Total cycles: {cycle_count}")
        self.robot_interface.close()

if __name__ == "__main__":
    gripper_test = GripperTest()
    
    # Run test for 60 seconds (6 complete open/close cycles)
    gripper_test.run_gripper_test(duration_seconds=60) 