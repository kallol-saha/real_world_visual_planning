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

from frankapanda.calibration.robot_controller import FrankaOSCController
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
    cam_id = 3
    # 0: right -     000880595012
    # 2: left -     000059793721
    # 1: front -    000180921812
    # 3: back -     000263392612
    initial_joint_positions = {
        0: [-0.9815314609720331, 0.31031684451395847, 0.630475446598282, -1.823140417152359, -0.14117339535320805, 2.020006045755246, -1.9992902488784574],
        1: [-0.83424677, 0.42084166, 0.2774182, -1.97982254, -0.1749291, 2.40231471, 0.27310384],
        2: [-0.81592058, 0.39429853, 0.29050235, -1.88333403, -0.17686262, 2.28619198, 1.98916667],
        3: [-0.85456277, 0.36942704, 0.38232294, -1.88742087, -0.45677587, 2.19400042, -2.88310376]
        # 3: [ 1.9666895  -0.58094072 -2.34559199 -1.91077682 -0.83849069  2.12780669 2.84485489]
    }[cam_id]
    
    # Perform the movements and record data
    move_robot_and_record_data(
        cam_id=cam_id, num_movements=50, debug=False, 
        initial_joint_positions=initial_joint_positions)
    

if __name__ == "__main__":
    main()