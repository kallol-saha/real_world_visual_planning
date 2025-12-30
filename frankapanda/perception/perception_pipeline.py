"""
Perception Pipeline Orchestrator

This script:
1. Runs camera capture scripts sequentially (cam_0 then cam_1)
2. Receives point clouds via ZMQ
3. Combines point clouds
4. Applies spatial bounds filtering
5. Performs FPS downsampling
6. Publishes final point cloud via ZMQ for downstream consumers

Supports both single-shot and continuous modes.
"""

import argparse
import subprocess
import numpy as np
import zmq
import pickle
import open3d as o3d
import time
import signal
import sys
from pathlib import Path

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n\nShutdown requested... cleaning up")
    shutdown_requested = True


def apply_spatial_bounds(pcd, rgb, bounds):
    """
    Apply spatial bounds filtering to point cloud.

    Args:
        pcd: Nx3 point cloud
        rgb: Nx3 RGB colors
        bounds: dict with 'x', 'y', 'z' keys containing [min, max] lists

    Returns:
        filtered_pcd, filtered_rgb
    """
    mask = np.ones(len(pcd), dtype=bool)

    # Apply bounds on each axis
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        if axis_name in bounds:
            min_val, max_val = bounds[axis_name]
            axis_mask = (pcd[:, axis_idx] >= min_val) & (pcd[:, axis_idx] <= max_val)
            mask &= axis_mask

    return pcd[mask], rgb[mask]


def fps_downsample(pcd, rgb, num_points):
    """
    Downsample point cloud using Farthest Point Sampling (FPS).

    Args:
        pcd: Nx3 point cloud
        rgb: Nx3 RGB colors
        num_points: Target number of points

    Returns:
        downsampled_pcd, downsampled_rgb
    """
    if len(pcd) <= num_points:
        print(f"Point cloud has {len(pcd)} points, no downsampling needed")
        return pcd, rgb

    # Convert to Open3D point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb)

    # FPS downsampling
    print(f"Downsampling from {len(pcd)} to {num_points} points using FPS...")
    pcd_downsampled = pcd_o3d.farthest_point_down_sample(num_points)

    # Convert back to numpy
    points = np.asarray(pcd_downsampled.points)
    colors = np.asarray(pcd_downsampled.colors)

    return points, colors


def capture_camera(cam_id, zmq_port):
    """
    Run camera capture subprocess and return success status.

    Args:
        cam_id: Camera ID (0 or 1)
        zmq_port: ZMQ port for communication

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Starting capture for Camera {cam_id}")
    print(f"{'='*60}")

    script_path = Path(__file__).parent / "capture_single_camera.py"

    # Run as subprocess
    result = subprocess.run(
        ['python', str(script_path), '--cam_id', str(cam_id), '--zmq_port', str(zmq_port)],
        capture_output=True,
        text=True
    )

    # Print subprocess output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"ERROR: Camera {cam_id} capture failed with return code {result.returncode}")
        return False

    return True


def run_pipeline_iteration(receiver, publisher, bounds, num_points, receive_port, save=False, iteration=None):
    """
    Run one iteration of the perception pipeline.

    Args:
        receiver: ZMQ receiver socket
        publisher: ZMQ publisher socket
        bounds: Spatial bounds dictionary
        num_points: Target number of points for downsampling
        receive_port: Port for receiving camera data
        save: Whether to save output to file
        iteration: Iteration number (for logging in continuous mode)

    Returns:
        True if successful, False otherwise
    """
    iter_prefix = f"[Iteration {iteration}] " if iteration is not None else ""

    # Capture from both cameras sequentially
    camera_data = {}

    for cam_id in [0, 1]:
        # Launch camera capture subprocess
        success = capture_camera(cam_id, receive_port)

        if not success:
            print(f"{iter_prefix}ERROR: Failed to capture from camera {cam_id}")
            return False

        # Receive data via ZMQ
        print(f"\n{iter_prefix}Waiting to receive data from Camera {cam_id}...")
        data = pickle.loads(receiver.recv())
        camera_data[cam_id] = data
        print(f"{iter_prefix}Received {len(data['pcd'])} points from Camera {cam_id}")

    # Combine point clouds from both cameras
    print("\n" + "="*60)
    print(f"{iter_prefix}COMBINING POINT CLOUDS")
    print("="*60)

    pcd_combined = np.vstack([camera_data[0]['pcd'], camera_data[1]['pcd']])
    rgb_combined = np.vstack([camera_data[0]['rgb'], camera_data[1]['rgb']])

    print(f"{iter_prefix}Combined point cloud: {len(pcd_combined)} points")

    # Apply spatial bounds filtering
    print("\n" + "="*60)
    print(f"{iter_prefix}APPLYING SPATIAL BOUNDS")
    print("="*60)
    print(f"{iter_prefix}Bounds: {bounds}")

    pcd_filtered, rgb_filtered = apply_spatial_bounds(pcd_combined, rgb_combined, bounds)
    print(f"{iter_prefix}After bounds filtering: {len(pcd_filtered)} points")

    # FPS downsampling
    print("\n" + "="*60)
    print(f"{iter_prefix}FPS DOWNSAMPLING")
    print("="*60)

    pcd_final, rgb_final = fps_downsample(pcd_filtered, rgb_filtered, num_points)
    print(f"{iter_prefix}Final point cloud: {len(pcd_final)} points")

    # Prepare final data
    final_data = {
        'pcd': pcd_final,
        'rgb': rgb_final,
        'num_points': len(pcd_final),
        'bounds': bounds
    }

    # Publish final point cloud
    print("\n" + "="*60)
    print(f"{iter_prefix}PUBLISHING FINAL POINT CLOUD")
    print("="*60)

    publisher.send(pickle.dumps(final_data))
    print(f"{iter_prefix}Published final point cloud")

    # Optional: Save to file
    if save:
        output_dir = Path(__file__).parent / "data" / "perception_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "final_pcd.npy", pcd_final)
        np.save(output_dir / "final_rgb.npy", rgb_final)
        print(f"\n{iter_prefix}Saved to {output_dir}/")

    return True


def main():
    global shutdown_requested

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Perception Pipeline')
    parser.add_argument('--receive_port', type=int, default=6555,
                        help='ZMQ port to receive camera data (default: 6555)')
    parser.add_argument('--publish_port', type=int, default=6556,
                        help='ZMQ port to publish final point cloud (default: 6556)')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of points for FPS downsampling (default: 4096)')
    parser.add_argument('--save', action='store_true',
                        help='Save final point cloud to data/perception_output/')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously in a loop (default: single-shot)')
    parser.add_argument('--rate', type=float, default=1.0,
                        help='Loop rate in Hz for continuous mode (default: 1.0)')
    args = parser.parse_args()

    mode_str = "CONTINUOUS" if args.continuous else "SINGLE-SHOT"
    print("="*60)
    print(f"PERCEPTION PIPELINE STARTING ({mode_str} MODE)")
    print("="*60)

    if args.continuous:
        print(f"Loop rate: {args.rate} Hz (period: {1.0/args.rate:.2f}s)")
        print("Press Ctrl+C to stop\n")

    # Define spatial bounds
    bounds = {
        'x': [0.2, 0.8],
        'y': [-0.5, 0.5],
        'z': [-0.1, 0.7]
    }

    # Setup ZMQ receiver socket (PULL mode)
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{args.receive_port}")

    # Setup ZMQ publisher socket (PUB mode)
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{args.publish_port}")

    print(f"ZMQ Configuration:")
    print(f"  Receiving camera data on port: {args.receive_port}")
    print(f"  Publishing final point cloud on port: {args.publish_port}\n")

    try:
        if args.continuous:
            # Continuous mode - loop until shutdown requested
            iteration = 1
            loop_period = 1.0 / args.rate

            while not shutdown_requested:
                start_time = time.time()

                print("\n" + "="*60)
                print(f"ITERATION {iteration} START")
                print("="*60 + "\n")

                success = run_pipeline_iteration(
                    receiver, publisher, bounds, args.num_points,
                    args.receive_port, args.save, iteration
                )

                if not success:
                    print(f"\nIteration {iteration} failed, stopping continuous mode")
                    break

                elapsed = time.time() - start_time
                print(f"\nIteration {iteration} completed in {elapsed:.2f}s")

                # Sleep to maintain desired rate
                sleep_time = loop_period - elapsed
                if sleep_time > 0:
                    print(f"Sleeping for {sleep_time:.2f}s to maintain {args.rate} Hz rate...")
                    time.sleep(sleep_time)
                else:
                    print(f"WARNING: Iteration took longer ({elapsed:.2f}s) than loop period ({loop_period:.2f}s)")

                iteration += 1

            print("\n" + "="*60)
            print(f"CONTINUOUS MODE STOPPED (ran {iteration-1} iterations)")
            print("="*60)

        else:
            # Single-shot mode - run once
            success = run_pipeline_iteration(
                receiver, publisher, bounds, args.num_points,
                args.receive_port, args.save
            )

            if success:
                print("\n" + "="*60)
                print("PERCEPTION PIPELINE COMPLETE")
                print("="*60)

    finally:
        # Clean up
        print("\nCleaning up ZMQ sockets...")
        receiver.close()
        publisher.close()
        context.term()
        print("Shutdown complete")


if __name__ == '__main__':
    main()
