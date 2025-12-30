"""
PerceptionPipeline class wrapper for easy programmatic access to the perception system.
"""

import zmq
import pickle
import numpy as np
from typing import Optional, Tuple, Dict
import threading
import time


class PerceptionPipeline:
    """
    High-level wrapper for the dual camera perception pipeline.

    Allows programmatic access to perception data without needing to run
    separate processes or deal with ZMQ directly.
    """

    def __init__(
        self,
        publish_port: int = 6556,
        timeout_ms: int = 60000
    ):
        """
        Initialize the perception pipeline client.

        Args:
            publish_port: ZMQ port to subscribe to for final point clouds
            timeout_ms: Timeout for receiving data in milliseconds
        """
        self.publish_port = publish_port
        self.timeout_ms = timeout_ms

        # Setup ZMQ subscriber
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.publish_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

        self._latest_data = None
        self._running = False
        self._thread = None

    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the latest point cloud from the perception pipeline.

        This is a blocking call that waits for the next published point cloud.

        Returns:
            Tuple of (pcd, rgb) where:
                pcd: Nx3 array of XYZ coordinates in meters
                rgb: Nx3 array of RGB colors in [0, 1]

        Raises:
            TimeoutError: If no data received within timeout period
        """
        try:
            data = pickle.loads(self.socket.recv())
            return data['pcd'], data['rgb']
        except zmq.Again:
            raise TimeoutError(f"No data received within {self.timeout_ms}ms")

    def get_point_cloud_dict(self) -> Dict:
        """
        Get the latest point cloud data as a dictionary.

        Returns:
            Dictionary containing:
                - 'pcd': Nx3 point cloud
                - 'rgb': Nx3 RGB colors
                - 'num_points': Number of points
                - 'bounds': Bounds used for filtering

        Raises:
            TimeoutError: If no data received within timeout period
        """
        try:
            data = pickle.loads(self.socket.recv())
            return data
        except zmq.Again:
            raise TimeoutError(f"No data received within {self.timeout_ms}ms")

    def start_continuous_listener(self):
        """
        Start a background thread that continuously listens for point clouds.

        Use get_latest() to retrieve the most recent point cloud without blocking.
        """
        if self._running:
            print("Listener already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self):
        """Background loop that continuously receives point clouds."""
        while self._running:
            try:
                data = pickle.loads(self.socket.recv())
                self._latest_data = data
            except zmq.Again:
                # Timeout, continue waiting
                pass
            except Exception as e:
                print(f"Error in listen loop: {e}")
                break

    def get_latest(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the most recently received point cloud (non-blocking).

        Returns None if no data has been received yet.

        Returns:
            Tuple of (pcd, rgb) or None
        """
        if self._latest_data is None:
            return None
        return self._latest_data['pcd'], self._latest_data['rgb']

    def get_latest_dict(self) -> Optional[Dict]:
        """
        Get the most recently received point cloud data as dictionary (non-blocking).

        Returns None if no data has been received yet.
        """
        return self._latest_data

    def stop_continuous_listener(self):
        """Stop the background listener thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self):
        """Clean up resources."""
        self.stop_continuous_listener()
        self.socket.close()
        self.context.term()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass
