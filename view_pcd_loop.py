"""
Continuously capture the latest point cloud + segmentation from the perception
pipeline and print the gray-tray midpoint + xy extents on every frame.

Visualizations are commented out and there is no user prompt — the loop runs
indefinitely (Ctrl+C to stop).

Prereq:
    python -m frankapanda.perception.perception_pipeline --continuous --segment \
        --seg_labels "cup. bowl. plate. gray tray."
"""

import numpy as np
import zmq

from robo_utils.visualization.plotting import plot_pcd  # noqa: F401 (kept for re-enable)
from frankapanda.perception import PerceptionPipeline


PUBLISH_PORT = 1235
TIMEOUT_MS = 10000
TRAY_LABEL = "gray tray"


def _drain(perception: PerceptionPipeline):
    """Drop buffered ZMQ SUB msgs so the next recv is fresh."""
    dropped = 0
    while True:
        try:
            perception.socket.recv(flags=zmq.DONTWAIT)
            dropped += 1
        except zmq.Again:
            break
    if dropped:
        print(f"  drained {dropped} stale frames.")


def main():
    print(f"Connecting to perception pipeline on port {PUBLISH_PORT}")
    perception = PerceptionPipeline(publish_port=PUBLISH_PORT, timeout_ms=TIMEOUT_MS)

    try:
        i = 0
        while True:
            i += 1
            _drain(perception)
            try:
                pdata = perception.get_point_cloud_dict()
            except TimeoutError:
                print(f"[{i}] timeout — retrying.")
                continue
            pcd_np = pdata["pcd"]
            rgb_np = pdata.get("rgb")
            seg_labels = pdata.get("seg_labels")
            seg_label_names = pdata.get("seg_label_names")

            # --- visualization (disabled) ---
            # if rgb_np is not None:
            #     plot_pcd(pcd_np, rgb_np, base_frame=True)
            # else:
            #     plot_pcd(pcd_np, base_frame=True)
            # if seg_labels is not None and seg_label_names is not None:
            #     seg_labels_a = np.asarray(seg_labels)
            #     for k, name in enumerate(seg_label_names):
            #         mask = (seg_labels_a == k)
            #         n = int(mask.sum())
            #         if n == 0:
            #             continue
            #         sub_pcd = pcd_np[mask]
            #         sub_rgb = rgb_np[mask] if rgb_np is not None else None
            #         if sub_rgb is not None:
            #             plot_pcd(sub_pcd, sub_rgb, base_frame=True)
            #         else:
            #             plot_pcd(sub_pcd, base_frame=True)
            #     bg_mask = (seg_labels_a == -1)
            #     n_bg = int(bg_mask.sum())
            #     if n_bg > 0:
            #         sub_pcd = pcd_np[bg_mask]
            #         sub_rgb = rgb_np[bg_mask] if rgb_np is not None else None
            #         if sub_rgb is not None:
            #             plot_pcd(sub_pcd, sub_rgb, base_frame=True)
            #         else:
            #             plot_pcd(sub_pcd, base_frame=True)

            # --- tray midpoint + xy extents ---
            if seg_labels is None or seg_label_names is None:
                print(f"[{i}] pcd {pcd_np.shape}  no seg_labels in payload "
                      f"(run perception_pipeline with --segment).")
                continue
            seg_labels_a = np.asarray(seg_labels)
            if TRAY_LABEL not in seg_label_names:
                print(f"[{i}] pcd {pcd_np.shape}  TRAY_LABEL '{TRAY_LABEL}' "
                      f"not in {list(seg_label_names)}.")
                continue
            tray_idx = list(seg_label_names).index(TRAY_LABEL)
            tray_mask = (seg_labels_a == tray_idx)
            n_tray = int(tray_mask.sum())
            if n_tray == 0:
                print(f"[{i}] pcd {pcd_np.shape}  tray '{TRAY_LABEL}' "
                      f"idx={tray_idx}  pts=0 (no tray detected this frame).")
                continue
            tray_pts = pcd_np[tray_mask]
            mid = tray_pts.mean(axis=0)                                   # (3,)
            x_min, x_max = float(tray_pts[:, 0].min()), float(tray_pts[:, 0].max())
            y_min, y_max = float(tray_pts[:, 1].min()), float(tray_pts[:, 1].max())
            x_extent = x_max - x_min
            y_extent = y_max - y_min
            print(f"[{i}] tray pts={n_tray}  "
                  f"midpoint=({mid[0]:.4f}, {mid[1]:.4f}, {mid[2]:.4f})  "
                  f"x=[{x_min:.4f}, {x_max:.4f}] (ext {x_extent:.4f})  "
                  f"y=[{y_min:.4f}, {y_max:.4f}] (ext {y_extent:.4f})")
    finally:
        perception.close()


if __name__ == "__main__":
    main()
