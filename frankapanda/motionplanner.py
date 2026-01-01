"""
We do motion planning using Curobo.
"""

import os
import gc
import copy

# Third Party
from sympy.printing.latex import true
import torch
import numpy as np
import trimesh
from trimesh.creation import icosphere, axis
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.types import Cuboid, WorldConfig, Mesh, VoxelGrid
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# Local
from robo_utils.conversion_utils import (
    pose_to_transformation,
    move_pose_along_local_z,
    transformation_to_pose,
    invert_transformation
)
# from visplan.utils import to_torch_pose

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EE_LINK_CENTER_TO_GRIPPER_TIP = 0.13

# Constants for motion planning
CUROBO_ASSETS_PATH = "visplan/submodules/curobo/src/curobo/content/assets/"       # Have to save here because cuRobo looks for mesh obstacles here
POST_GRASP_LIFT = 0.15
GRASP_DEPTH = 0.25

class MotionPlanner:

    def __init__(
        self,
        pointcloud: np.ndarray
        ):
        """
        Initialize the motion planner with point cloud collision checking.

        Args:
            voxel_size: Size of each voxel in meters (default 0.02 = 2cm).
            voxel_dims: Dimensions of the voxel grid in meters [x, y, z]. Default is [2.0, 2.0, 2.0].
            voxel_pose: Pose of the voxel grid center [x, y, z, qw, qx, qy, qz]. Default is [0, 0, 0, 1, 0, 0, 0].
        """
        
        self.pointcloud = pointcloud
        self.reset_planner(pointcloud)

        self.along_z_axis_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 1, 0], device="cuda:0"),
            project_to_goal_frame=True  # with respect to the goal frame
        )
        self.along_z_axis_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.along_z_axis_constraint,          
        )

        self.lift_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 1, 1, 0], device="cuda:0"),
            project_to_goal_frame=False  # with respect to the world frame
        )
        self.lift_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.lift_constraint,          
        )

        self.only_rotation_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([0, 0, 0, 0, 1, 1], device="cuda:0"),
            project_to_goal_frame=False  # with respect to the world frame
        )
        self.only_rotation_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.only_rotation_constraint,          
        )

        self.only_xy_translation_constraint = PoseCostMetric(
            hold_vec_weight = torch.tensor([1, 1, 1, 0, 0, 1], device="cuda:0"),
            project_to_goal_frame=False  # with respect to the world frame
        )
        self.only_xy_translation_plan_config = MotionGenPlanConfig(
            max_attempts=100,
            pose_cost_metric=self.only_xy_translation_constraint,          
        )

        print("")

        self.links = [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
            "attached_object",
        ]
        
    def reset_planner(self, pointcloud: np.ndarray):

        print("Building CuRobo World")
        setup_curobo_logger("error")
        self.tensor_args = TensorDeviceType()
        robot_file = "franka.yml"

        # Create world configuration with voxel grid for point cloud collision
        self.world_config = WorldConfig()

        self.back_wall = Cuboid(
            name = "back_wall",
            pose = [-0.4, 0., 0.5, 1, 0, 0, 0],
            dims = [0.2, 1.4, 1.0]
        )

        self.table = Cuboid(
            name = "table",
            pose = [0.5, 0., -(-0.03 + 0.2), 1, 0, 0, 0],
            dims = [1., 1.4, 0.4]
        )

        self.front_wall = Cuboid(
            name = "front_wall",
            pose = [0.85, 0., 0.5, 1, 0, 0, 0],
            dims = [0.2, 1.4, 1.0]
        )

        # These are from robot perspective
        self.right_wall = Cuboid(
            name = "right_wall",
            pose = [0.5, -(0.52 + 0.2/2), 0.5, 1, 0, 0, 0],
            dims = [1., 0.2, 1.0]
        )

        self.pointcloud_mesh = Mesh.from_pointcloud(
            pointcloud,
            pitch=0.02,
            name="pointcloud_mesh",
            pose=[0, 0, 0, 1, 0, 0, 0],
            filter_close_points=0.3,
        )

        self.shelf_top = Cuboid(
            name = "table",
            pose = [0.56, 0.45, (0.58 + 0.1/2), 1, 0, 0, 0],
            dims = [0.7, 0.7, 0.1]
        )

        self.world_config.add_obstacle(self.back_wall)
        self.world_config.add_obstacle(self.table)
        self.world_config.add_obstacle(self.front_wall)
        self.world_config.add_obstacle(self.pointcloud_mesh)
        self.world_config.add_obstacle(self.right_wall)
        self.world_config.add_obstacle(self.shelf_top)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            self.world_config,
            self.tensor_args,
            interpolation_dt=0.01,
            use_cuda_graph=False,
            interpolation_steps=10000,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        # Warmup with goalset planning since we use plan_grasp which requires goalset
        # Set n_goalset to the maximum number of grasps you expect to use (Basically the maximum number of poses that can be passed to plan_goalset)

        # self.motion_gen.warmup(enable_graph=True, n_goalset=100)
        self.motion_gen.warmup(n_goalset=200)
       
    def set_collision_world_components(
        self,  
        enable: bool,
        objects: list[str] = [], 
        collision_links: list[str] = []
        ):
        """
        Enable or disable collision checking for world obstacles and robot links.
        
        Args:
            enable: If True, enable collision checking; if False, disable it.
            objects: List of obstacle names to enable/disable.
            collision_links: List of robot link names to enable/disable collision for.
        """
        if len(collision_links) > 0:
            self.motion_gen.toggle_link_collision(collision_links, enable)
            # Enable all other links that are not in the collision_links list
            other_links = [link for link in self.links if link not in collision_links]
            if len(other_links) > 0:
                self.motion_gen.toggle_link_collision(other_links, True)

        for object_name in objects:
            self.motion_gen.world_coll_checker.enable_obstacle(enable=enable, name=object_name)
    
    def clear_gpu_memory(self):
        """
        Clear GPU memory cache. Call this after planning operations to free up GPU memory.
        This helps prevent out-of-memory errors when doing multiple planning operations.
        """
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def update_pointcloud_collision_world(
        self,
        pointcloud: np.ndarray,
        filter_close_distance: float = 0.0,
        ):
        """
        Update the collision world with a point cloud.

        The point cloud should already be in the robot base frame. This method converts the
        point cloud to a voxel grid with ESDF (Euclidean Signed Distance Field) for efficient
        collision checking.

        Args:
            pointcloud: Point cloud as a numpy array of shape (N, 3) representing [x, y, z]
                        coordinates in the robot base frame.
            filter_close_distance: Filter out points closer than this distance to the origin.
                                   Useful for removing points that are too close to the robot base.
        """
        # Filter points if requested
        if filter_close_distance > 0.0:
            dist = np.linalg.norm(pointcloud, axis=-1)
            pointcloud = pointcloud[dist > filter_close_distance]

        # Convert point cloud to mesh using marching cubes
        # This creates a watertight mesh from the point cloud
        mesh_from_pc = Mesh.from_pointcloud(
            pointcloud,
            pitch=self.voxel_size,
            name="pointcloud_mesh",
            pose=[0, 0, 0, 1, 0, 0, 0],
        )

        # Create a temporary world config with just the mesh
        temp_world = WorldConfig(mesh=[mesh_from_pc])

        # Create a collision checker to compute ESDF from the mesh
        temp_collision_config = WorldCollisionConfig(
            tensor_args=self.tensor_args,
            world_model=temp_world,
            checker_type=CollisionCheckerType.MESH,
            max_distance=0.1,
        )
        temp_collision_checker = create_collision_checker(temp_collision_config)

        # Compute ESDF in the bounding box defined by the voxel grid
        esdf = temp_collision_checker.get_esdf_in_bounding_box(
            Cuboid(
                name="pointcloud_voxel",
                pose=self.voxel_pose,
                dims=self.voxel_dims,
            ),
            voxel_size=self.voxel_size,
            dtype=torch.bfloat16,
        )

        # Update the motion gen's world collision checker with the new ESDF
        self.motion_gen.world_coll_checker.update_voxel_data(esdf)

        # Reset the graph planner buffer since the world has changed
        self.motion_gen.graph_planner.reset_buffer()

        # Clean up temporary collision checker
        del temp_collision_checker
        self.clear_gpu_memory()

    def update_pointcloud_collision_world_direct(
        self,
        pointcloud: np.ndarray,
        filter_close_distance: float = 0.0,
        ):
        """
        Update the collision world with a point cloud using direct voxelization.

        This is a faster alternative to update_pointcloud_collision_world that directly
        voxelizes the point cloud without creating a mesh first. However, this may be
        less accurate for thin structures since it only marks occupied voxels.

        Args:
            pointcloud: Point cloud as a numpy array of shape (N, 3) representing [x, y, z]
                        coordinates in the robot base frame.
            filter_close_distance: Filter out points closer than this distance to the origin.
        """
        # Filter points if requested
        if filter_close_distance > 0.0:
            dist = np.linalg.norm(pointcloud, axis=-1)
            pointcloud = pointcloud[dist > filter_close_distance]

        # Convert to torch tensor
        points_tensor = torch.tensor(pointcloud, dtype=torch.float32, device="cuda:0")

        # Get voxel grid shape info
        voxel_grid_shape = self.voxel_grid.get_grid_shape()
        grid_shape = voxel_grid_shape[0]  # [nx, ny, nz]
        low = voxel_grid_shape[1]  # [-dims[0]/2, -dims[1]/2, -dims[2]/2] relative to pose

        # Transform points into the voxel grid's local frame
        # The voxel grid's pose is [x, y, z, qw, qx, qy, qz]
        voxel_center = torch.tensor(self.voxel_pose[:3], device="cuda:0", dtype=torch.float32)
        points_local = points_tensor - voxel_center

        # Compute voxel indices for each point (relative to voxel grid's local frame)
        low_tensor = torch.tensor(low, device="cuda:0", dtype=torch.float32)
        voxel_indices = ((points_local - low_tensor) / self.voxel_size).long()

        # Filter points that are outside the voxel grid
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_shape[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_shape[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_shape[2])
        )
        voxel_indices = voxel_indices[valid_mask]

        # Create feature tensor (ESDF approximation using distance transform would be ideal,
        # but for simplicity we'll mark occupied voxels with positive distance)
        n_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
        features = torch.zeros(n_voxels, dtype=torch.bfloat16, device="cuda:0")
        features[:] = -100.0  # Mark all as free (negative = outside obstacle)

        # Mark occupied voxels
        if voxel_indices.shape[0] > 0:
            flat_indices = (
                voxel_indices[:, 0] * grid_shape[1] * grid_shape[2] +
                voxel_indices[:, 1] * grid_shape[2] +
                voxel_indices[:, 2]
            )
            features[flat_indices] = 0.1  # Mark as inside obstacle (positive = inside)

        # Create updated voxel grid
        updated_voxel = VoxelGrid(
            name="pointcloud_voxel",
            dims=self.voxel_dims,
            pose=self.voxel_pose,
            voxel_size=self.voxel_size,
            feature_tensor=features.view(1, -1),
            feature_dtype=torch.bfloat16,
        )

        # Update the world collision checker
        self.motion_gen.world_coll_checker.update_voxel_data(updated_voxel)

        # Reset the graph planner buffer since the world has changed
        self.motion_gen.graph_planner.reset_buffer()

    def plan_to_joint_state(self, current_joint_state: torch.Tensor, goal_joint_state: torch.Tensor, holding_object: bool = False):
        """
        Plan to a joint state.
        """
        
        start_joint = JointState.from_position(current_joint_state.view(1, -1))
        goal_joint = JointState.from_position(goal_joint_state.view(1, -1))
        
        plan_config = MotionGenPlanConfig(
            max_attempts=100,
        )

        if not holding_object:
            disable_collision_links = ["attached_object"]
            self.motion_gen.toggle_link_collision(disable_collision_links, False)

        result = self.motion_gen.plan_single_js(start_joint, goal_joint, plan_config)
        self.clear_gpu_memory()

        success = result.success.item()
        if success:
            return result.get_interpolated_plan().position, success
        else:
            return None, success
    
    # ==================================================== #

    # ================ PARALLEL PLANNING FUNCTIONS MULTI OBJECT ================ #

    def inverse_kinematics(
        self, 
        input_grasp_poses: torch.Tensor, 
        objects_to_ignore: list[str] = [], 
        disable_collision_links: list[str] = ["attached_object"]
        ):
        """
        Parallel IK solver for multiple grasp poses.
        
        Note: Maximum batch size is 1000. Input batch size should not exceed this limit.
        
        Args:
            input_grasp_poses: Grasp poses to reach ---> (N, 7) torch tensor, (x, y, z, qw, qx, qy, qz)
                where N <= max batch size that fits on the GPU (for RTX 4090, it is 1000)
            objects_to_ignore: List of object names to disable for collision checking during planning
            disable_collision_links: List of collision links to disable for collision checking during planning. 
            By default, it is ["attached_object"]
            # Collision link names:
            # [
            #     "panda_link0",
            #     "panda_link1",
            #     "panda_link2",
            #     "panda_link3",
            #     "panda_link4",
            #     "panda_link5",
            #     "panda_link6",
            #     "panda_link7",
            #     "panda_hand",
            #     "panda_leftfinger",
            #     "panda_rightfinger",
            #     "attached_object",
            # ]

        Returns:
            Valid grasp poses and joint states.
        """

        # Disable collision checking for relevant world components
        self.set_collision_world_components(
            enable=False, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )
        
        # Predefine the output joint states tensor:
        ik_solutions = torch.zeros((input_grasp_poses.shape[0], 7), device=input_grasp_poses.device, dtype=torch.float32)
        
        # Convert to Pose
        ik_poses = Pose(position=input_grasp_poses[..., :3], quaternion=input_grasp_poses[..., 3:])
        
        # Solve IK for the batch
        ik_result = self.motion_gen.ik_solver.solve_batch(ik_poses)
        self.clear_gpu_memory()

        # Get successful indices and solutions, then store them in the output tensor:
        success = ik_result.success.to(input_grasp_poses.device)    # (batch_size,) boolean tensor
        valid_indices = torch.where(success)[0]
        if len(valid_indices) > 0:
            ik_solutions[valid_indices] = ik_result.solution[valid_indices, 0]
        else:
            print(f"No valid IK solutions found for the poses")

        # Re-enable collision checking for relevant world components
        self.set_collision_world_components(
            enable=True, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )
            
        return ik_solutions, success
    
    def plan_to_goal_poses(
        self, 
        current_joints: torch.Tensor, 
        goal_poses: torch.Tensor, 
        objects_to_ignore: list[str] = [], 
        disable_collision_links: list[str] = ["attached_object"],
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(max_attempts=100)
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Plan a grasp using CuRobo.

        Note: Maximum batch size is 50. Input batch size should not exceed this limit.

        Args:
            current_joints: Current joint state of the robot ---> (N, 7) torch tensor where N <= 50
            goal_poses: Goal poses of the robot ---> (N, 7) torch tensor where N <= 50
            holding_object: Whether the robot is holding an object
            return_missing: If True, returns full-size trajectories (N, T, 7) with missing indices
                          filled with repeated current_joints. If False, returns only successful trajectories.
            objects_to_ignore: List of object names to disable for collision checking during planning

        Returns:
            trajectories: (N_success, T, 7) if return_missing=False, or (N, T, 7) if return_missing=True
            successful_indices: (N_success,) tensor of successful indices
        """
        
        traj_len = self.motion_gen.trajopt_solver.traj_tsteps
        assert current_joints.shape[0] == goal_poses.shape[0], "Current joint states and goal poses must have the same batch size"
        
        # Disable collision checking for relevant world components
        self.set_collision_world_components(
            enable=False, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )
        
        # Convert to JointState and Pose
        start_states = JointState.from_position(current_joints)
        curobo_goal_poses = Pose(position=goal_poses[..., :3], quaternion=goal_poses[..., 3:])

        # Initialize the solution, by default as just staying at the current joint state:
        trajectories = current_joints.clone()
        trajectories = trajectories.unsqueeze(1).repeat(1, traj_len, 1)  # (N, traj_len, 7)
        
        # NOTE: For cuRobo motion planning, 
        # use interpolated_plan if you need fine-grained waypoints, or optimized_plan if coarse is sufficient
        # Plan for the batch if environments:
        plan_result = self.motion_gen.plan_batch(
            start_state=start_states, 
            goal_pose=curobo_goal_poses, 
            plan_config=plan_config
        )
        # self.clear_gpu_memory()

        # Get successful indices and solutions, then store them in the output tensor:
        success = plan_result.success.to(current_joints.device)    # (batch_size,) boolean tensor
        successful_indices = torch.where(success)[0]

        if len(successful_indices) == 0:
            return trajectories, success

        if plan_result.optimized_plan.position.ndim == 2:   # Account for batch size of 1
            optimized_plan = plan_result.optimized_plan.position.unsqueeze(0).clone()
        else:
            optimized_plan = plan_result.optimized_plan.position.clone()
        trajectories[successful_indices] = optimized_plan[successful_indices].clone()

        # Re-enable collision checking for relevant world components
        self.set_collision_world_components(
            enable=True, 
            objects=objects_to_ignore, 
            collision_links=disable_collision_links
        )

        return trajectories, success
    
    # ==================================================== #

    def check_pose_collision(self, pose: torch.Tensor):
        """
        Check if a pose is collision free.
        """
        
        pose = Pose(pose[:3], quaternion=pose[3:])
        result = self.motion_gen.ik_solver.solve_single(pose)
        return result.js_solution.position[0], result.success.item()
    
    def attach_objects_to_robot(self, current_joint_state: torch.Tensor, object_names: list):
        """
        Attach objects to the robot.
        """
        joint_state = JointState.from_position(current_joint_state.view(1, -1))
        success = self.motion_gen.attach_objects_to_robot(
            joint_state, 
            object_names,
            sphere_fit_type = SphereFitType.VOXEL_SURFACE,
            surface_sphere_radius=0.03)

        return success
    
    def detach_objects_from_robot(self):
        """
        Detach objects from the robot.
        """
        success = self.motion_gen.detach_object_from_robot()
        return success
    
    def visualize_world_and_robot(self, q: torch.Tensor = None, pose: torch.Tensor = None):

        """
        Args:
            q: Joint configuration tensor. Shape (7,) or (B, 7).
            pose: Pose tensor. Shape (7,) or (B, 7).
            env_idx: Environment index.
        """

        # Optional: visualize an arbitrary pose as a frame in the scene
        T = None
        if pose is not None:
            if isinstance(pose, torch.Tensor):
                pose_np = pose.detach().cpu().numpy()
            else:
                pose_np = np.asarray(pose, dtype=np.float32)
            # pose_np expected as (x, y, z, qw, qx, qy, qz)
            T = pose_to_transformation(pose_np, format='wxyz')
        
        world = self.world_config
        scene = WorldConfig.get_scene_graph(world)

        # Add a small axis triad to visualize the provided pose
        if T is not None:
            frame = axis(origin_size=0.02, axis_length=0.12, transform=T)
            scene.add_geometry(frame)

        # robot spheres from FK
        if q is None:
            q = torch.zeros(7, dtype=torch.float32, device="cuda:0")
        if q.dim() == 1:
            q = q.view(1, -1)
        kin = self.motion_gen.compute_kinematics(JointState.from_position(q))
        spheres = kin.robot_spheres.squeeze(0).cpu().numpy()  # [n,4] x,y,z,r

        for x, y, z, r in spheres:
            s = icosphere(subdivisions=2, radius=float(r))
            s.apply_translation([float(x), float(y), float(z)])
            s.visual.face_colors = [200, 50, 50, 120]
            scene.add_geometry(s)

        scene.show()
    
    def fk(self, q: torch.Tensor, link_name: str = None):
        """
        Forward kinematics using CuRobo via MotionGen's kinematics.

        Args:
            q: Joint configuration tensor. Shape (7,) or (B, 7).
            link_name: Optional link name. If None, returns end-effector pose.

        Returns:
            CudaRobotModelState with fields like ee_pos [B, 3], ee_rot [B, 4] (wxyz),
            link_pos, link_rot, etc.
        """
        if q.dim() == 1:
            q = q.view(1, -1)
        ee_pose = torch.zeros_like(q)
        fk_result = self.motion_gen.ik_solver.kinematics.get_state(q, link_name=link_name)
        ee_pose[:, :3] = fk_result.ee_position
        ee_pose[:, 3:] = fk_result.ee_quaternion
        if q.shape[0] == 1:
            ee_pose = ee_pose.squeeze(0)
        return ee_pose
