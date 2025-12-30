import numpy as np
import cv2
import pickle, os
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation

def estimate_tag_pose(finger_pose, t_tag_to_hand_base):
    """
    Estimate the tag pose given the gripper pose by applying the gripper-to-tag transformation.

    Args:
        finger_pose (eef_pose): 4x4 transformation matrix from gripper to robot base
        t_tag_to_hand_base: Base translation from tag to hand (before corner offset)
    Returns:
        hand_pose: 4x4 transformation matrix from hand to robot base
        tag_pose: 4x4 transformation matrix from tag to robot base
    """
    from scipy.spatial.transform import Rotation

    # Estimate the hand pose
    # finger_to_hand obtained from the product manual: 
    # [https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf]
    finger_to_hand = np.array([
        [0.707,  0.707, 0, 0],
        [-0.707, 0.707, 0, 0],
        [0, 0, 1, 0.1034],
        [0, 0, 0, 1],
    ])
    finger_to_hand = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.1034],
        [0, 0, 0, 1],
    ])
    hand_to_finger = np.linalg.inv(finger_to_hand)
    hand_pose = np.dot(finger_pose, hand_to_finger)

    t_tag_to_hand = t_tag_to_hand_base.copy()   # to tag corner
    t_tag_to_hand += np.array([0, -0.0275, 0.0275])         # to tag center
    # R_tag_to_hand = Rotation.from_quat([0.5, -0.5, 0.5, -0.5])
    R_tag_to_hand = Rotation.from_quat([0, 0, 0, 1])
    tag_to_hand = np.eye(4)
    tag_to_hand[:3, :3] = R_tag_to_hand.as_matrix()
    tag_to_hand[:3, 3] = t_tag_to_hand

    tag_pose = np.dot(hand_pose, tag_to_hand)
    
    return hand_pose, tag_pose

def solve_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def calculate_reprojection_error(tag_poses, target_poses, T_matrix):
    errors = []
    for tag_pose, target_pose in zip(tag_poses, target_poses):
        # Transform target pose using T_matrix
        transformed_target = np.dot(T_matrix, target_pose)
        transformed_pos = transformed_target[:3, 3]

        # Compare with tag pos
        tag_pos = tag_pose[:3, 3]
        error = np.linalg.norm(tag_pos - transformed_pos)
        errors.append(error)

    # Compute average error
    avg_error = np.mean(errors)
    return avg_error

def solve_extrinsic(gripper_poses, target_poses_in_camera, t_tag_to_hand_base, eye_to_hand=True):
    """
    Solve the extrinsic calibration between the camera and the base.
    """
    if eye_to_hand:
        # Calculate the transformation matrix from gripper to tag
        tag_poses = [estimate_tag_pose(pose, t_tag_to_hand_base)[1] for pose in gripper_poses]
    
    gripper_pos = np.array([pose[:3, 3] for pose in tag_poses])
    target_pos = np.array([pose[:3, 3] for pose in target_poses_in_camera])
    T = solve_rigid_transformation(target_pos, gripper_pos)

    # Calculate the reprojection error
    avg_error = calculate_reprojection_error(
        tag_poses, target_poses_in_camera, T)

    return T, avg_error


if __name__ == "__main__":
    # Load data
    cam_id = 1
    data_dirname = "data"
    data_filepath = os.path.join(data_dirname, f"cam{cam_id}_data.pkl")
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
    gripper_poses, target_poses_in_camera = zip(*data) 
    
    # Base t_tag_to_hand value (before corner offset)
    t_tag_to_hand_base = np.array([0.048914, 0.0275, 0.00753])
    
    # Define search range (interpolate between +/- values)
    # You can adjust these ranges and step sizes
    ranges = [
        np.linspace(t_tag_to_hand_base[0] - 0.01, t_tag_to_hand_base[0] + 0.01, 11),  # x: ±0.01 with 11 steps
        np.linspace(t_tag_to_hand_base[1] - 0.01, t_tag_to_hand_base[1] + 0.01, 11),  # y: ±0.01 with 11 steps
        np.linspace(t_tag_to_hand_base[2] - 0.01, t_tag_to_hand_base[2] + 0.01, 11),  # z: ±0.01 with 11 steps
    ]
    
    # Grid search over all combinations
    best_error = float('inf')
    best_t_tag_to_hand = None
    best_T = None
    total_combinations = len(ranges[0]) * len(ranges[1]) * len(ranges[2])
    current = 0
    
    print(f"Searching over {total_combinations} combinations...")
    print(f"Base t_tag_to_hand: {t_tag_to_hand_base}")
    print(f"Search ranges: x=[{ranges[0][0]:.6f}, {ranges[0][-1]:.6f}], "
          f"y=[{ranges[1][0]:.6f}, {ranges[1][-1]:.6f}], "
          f"z=[{ranges[2][0]:.6f}, {ranges[2][-1]:.6f}]")
    print()
    
    for x in ranges[0]:
        for y in ranges[1]:
            for z in ranges[2]:
                current += 1
                t_test = np.array([x, y, z])
                
                try:
                    T, error = solve_extrinsic(gripper_poses, target_poses_in_camera, t_test)
                    
                    if error < best_error:
                        best_error = error
                        best_t_tag_to_hand = t_test.copy()
                        best_T = T.copy()
                    
                    if current % 100 == 0:
                        print(f"Progress: {current}/{total_combinations} | Current best error: {best_error:.6f}")
                        
                except Exception as e:
                    print(f"Error with t_tag_to_hand={t_test}: {e}")
                    continue
    
    print("\n" + "="*60)
    print("GRID SEARCH RESULTS")
    print("="*60)
    print(f"Best t_tag_to_hand (before corner offset): {best_t_tag_to_hand}")
    print(f"Best calibration error: {best_error:.6f}")
    print(f"\nBest transformation matrix T:\n{best_T}")
    print("="*60)
    
    # Save the best calibration
    calib_dirname = os.path.join("data", "calibration_results")
    os.makedirs(calib_dirname, exist_ok=True)
    filepath = os.path.join(calib_dirname, f"cam{cam_id}_calibration_grid_search.npz")
    np.savez(filepath, T=best_T, t_tag_to_hand=best_t_tag_to_hand, error=best_error)

