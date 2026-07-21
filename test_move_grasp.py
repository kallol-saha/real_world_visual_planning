"""
Minimal move + grasp smoke test for FrankaPandaController.

No perception, no motion planner. Just:
  home -> open -> close (grasp) -> open

Select gripper mode:
    python test_move_grasp.py                 # robotiq (default)
    python test_move_grasp.py --mode franka   # franka hand

Prereq: robot connected + powered, deoxys running. For robotiq mode, the
Robotiq gripper must be plugged in. Run from repo root (configs/ paths are
relative). Keep e-stop within reach.
"""

import argparse

from frankapanda import FrankaPandaController
from frankapanda.controller import OPEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["robotiq", "franka"], default="robotiq")
    args = parser.parse_args()

    print(f">>> Building controller (gripper_mode={args.mode})")
    controller = FrankaPandaController(gripper_mode=args.mode)

    input("\nRobot will MOVE to home. Clear the workspace, then press Enter...")
    controller.move_to_joints(controller.home_joints, gripper_state=OPEN)
    print("  reached home.")

    input("\nPress Enter to OPEN gripper...")
    controller.open_gripper()
    print(f"  gripper state: {controller.get_gripper_state()} (OPEN=1, CLOSED=-1)")

    input("\nPlace an object between the fingers, then press Enter to CLOSE (grasp)...")
    controller.close_gripper()
    print(f"  gripper state: {controller.get_gripper_state()}")

    input("\nPress Enter to OPEN (release)...")
    controller.open_gripper()
    print(f"  gripper state: {controller.get_gripper_state()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
