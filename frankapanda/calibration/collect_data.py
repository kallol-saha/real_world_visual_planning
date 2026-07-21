"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import numpy as np
import os, pickle
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType

from robot_controller import FrankaOSCController
from marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation


def move_robot_and_record_data(
        cam_id,
        cam_type="kinect",
        num_movements=3, 
        debug=False,
        initial_joint_positions=None):
    """
    Move the robot to random poses and record the necessary data.
    """
    
    # Initialize the robot
    robot = FrankaOSCController(
        tip_offset=np.zeros(3),     # Set the default to 0 to disable accounting for the tip
    )

    # Initialize the camera
    k4a = PyK4A(device_id=cam_id)
    k4a.start()
    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)
    dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.DEPTH)

    data = []
    for _ in tqdm(range(num_movements)):
        # Generate a random target delta pose
        random_delta_pos = [np.random.uniform(-0.06, 0.06, size=(3,))]
        random_delta_axis_angle = [np.random.uniform(-0.5, 0.5, size=(3,))]
        robot.reset(joint_positions=initial_joint_positions)
        # import pdb; pdb.set_trace()
        robot.move_by(random_delta_pos, random_delta_axis_angle, num_steps=40, num_additional_steps=30)

        import time
        time.sleep(0.2)
        # Get current pose of the robot 
        gripper_pose = robot.eef_pose
        print(f"Gripper pos: {gripper_pose[:3, 3]}")

        # Capture IR frame from Kinect
        ir_frame = get_kinect_ir_frame(k4a)
        if ir_frame is not None:
            # Detect ArUco markers and get visualization
            corners, ids = detect_aruco_markers(ir_frame, debug=debug)


            # Estimate transformation if marker is detected
            if ids is not None and len(ids) > 0:
                print("\033[92m" + f"Detected {len(ids)} markers." + "\033[0m")
                transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
                if transform_matrix is not None:
                    data.append((
                        gripper_pose,       # gripper pose in base
                        transform_matrix    # tag pose in camera
                    ))
            else:
                print("\033[91m" + "No markers detected." + "\033[0m")
        else:
            print("\033[91m" + "No IR frame captured." + "\033[0m")
    
    print(f"Recorded {len(data)} data points.")
    
    # Save data
    os.makedirs("data", exist_ok=True)
    filepath = f"data/cam{cam_id}_data.pkl"
    with open(f"data/cam{cam_id}_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return filepath

def main():
    cam_id = 1
    # 0: front -     000880595012
    # 1: left -     000059793721
    initial_joint_positions = {
        0: [-1.12926786, 0.57204987, 0.57567669, -2.07868776, -0.14864151, 2.61236043, -0.02517276],
        1: [-0.74921682, 0.13623207, 0.37435664, -2.00871515, -0.54053575, 2.19774203, 2.34971468]
    }[cam_id]
    
    # Perform the movements and record data
    move_robot_and_record_data(
        cam_id=cam_id, num_movements=50, debug=False, 
        initial_joint_positions=initial_joint_positions)
    

if __name__ == "__main__":
    main()