"""
Example client script that receives the final point cloud from the perception pipeline.

This demonstrates how to subscribe to the perception pipeline output and use it
in downstream applications (planning, manipulation, etc.)

You can also use the PerceptionPipeline class from frankapanda.perception for easier access.
"""

import argparse
import numpy as np
from robo_utils.visualization.plotting import plot_pcd

# Import from frankapanda package (optional, but cleaner)
try:
    from frankapanda.perception import PerceptionPipeline
    USE_WRAPPER = True
except ImportError:
    # Fallback to direct ZMQ
    import zmq
    import pickle
    USE_WRAPPER = False


def main():
    parser = argparse.ArgumentParser(description='Perception Pipeline Client')
    parser.add_argument('--port', type=int, default=6556,
                        help='ZMQ port to receive final point cloud (default: 6556)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize received point cloud with Open3D')
    parser.add_argument('--timeout', type=int, default=60000,
                        help='Timeout in milliseconds (default: 60000)')
    args = parser.parse_args()

    print("="*60)
    print("PERCEPTION CLIENT")
    print("="*60)
    print(f"Subscribing to port: {args.port}")
    print(f"Timeout: {args.timeout}ms")

    if USE_WRAPPER:
        print("Using PerceptionPipeline wrapper class")
    else:
        print("Using direct ZMQ (wrapper not available)")

    print(f"Waiting for point cloud data...\n")

    try:
        if USE_WRAPPER:
            # Method 1: Using the PerceptionPipeline wrapper (recommended)
            with PerceptionPipeline(publish_port=args.port, timeout_ms=args.timeout) as perception:
                data = perception.get_point_cloud_dict()
        else:
            # Method 2: Direct ZMQ (fallback)
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect(f"tcp://localhost:{args.port}")
            socket.setsockopt(zmq.SUBSCRIBE, b'')
            socket.setsockopt(zmq.RCVTIMEO, args.timeout)
            data = pickle.loads(socket.recv())
            socket.close()
            context.term()

        print("="*60)
        print("RECEIVED POINT CLOUD DATA")
        print("="*60)

        pcd = data['pcd']
        rgb = data['rgb']
        num_points = data['num_points']
        bounds = data['bounds']

        print(f"Number of points: {num_points}")
        print(f"Point cloud shape: {pcd.shape}")
        print(f"RGB shape: {rgb.shape}")
        print(f"Bounds used: {bounds}")

        # Print statistics
        print("\nPoint cloud statistics:")
        print(f"  X: [{pcd[:, 0].min():.3f}, {pcd[:, 0].max():.3f}]")
        print(f"  Y: [{pcd[:, 1].min():.3f}, {pcd[:, 1].max():.3f}]")
        print(f"  Z: [{pcd[:, 2].min():.3f}, {pcd[:, 2].max():.3f}]")

        # Visualize
        print("\nLaunching visualization...")
        plot_pcd(pcd, rgb, base_frame=True)

        # Here you would use the point cloud for your application
        # For example:
        # - Send to motion planning algorithm
        # - Use for grasp planning
        # - Obstacle detection
        # etc.

        print("\n" + "="*60)
        print("CLIENT COMPLETE")
        print("="*60)

    except TimeoutError:
        print(f"\nERROR: Timeout after {args.timeout}ms")
        print("Make sure the perception pipeline is running!")
    except Exception as e:
        if not USE_WRAPPER and hasattr(e, '__class__') and e.__class__.__name__ == 'Again':
            print(f"\nERROR: Timeout after {args.timeout}ms")
            print("Make sure the perception pipeline is running!")
        else:
            raise


if __name__ == '__main__':
    main()
