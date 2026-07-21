# Dual Gripper Mode for FrankaPandaController

Date: 2026-07-20

## Goal

Add two selectable gripper modes to `frankapanda/controller.py`:

- `robotiq` (default) — current behavior, unchanged, byte-identical.
- `franka` — restore the pre-`5311d7f` Franka gripper, driven through the
  deoxys action byte.

Switching is done via a single constructor argument. When the argument is
left at its default, every existing caller (`FrankaPandaController()`) behaves
exactly as it does today.

## Non-goals

- No changes to any runner / caller code.
- No new config file or env var.
- No refactor of motion-planning or OSC logic beyond the gripper byte.

## Background

Before commit `5311d7f "smoothened motion"`, the gripper was the Franka hand,
commanded by appending a gripper byte to the deoxys action vector
(`[joints..., byte]`), streamed every control tick. State was read from
`robot_interface._gripper_state_buffer[-1].width`.

That commit swapped in a Robotiq gripper: a separate `pyrobotiqgripper` device
driven by its own `open()/close()`, with the deoxys byte sent as an inert
`GRIPPER_NOOP = 0.0`. State is read from `gripper.position_mm()`.

The Franka path still exists in git history and is recoverable from
`5311d7f^:frankapanda/controller.py`.

## Design

### Mode selection

```python
def __init__(self, gripper_mode: str = "robotiq"):
    assert gripper_mode in ("robotiq", "franka")
    self.gripper_mode = gripper_mode
```

Default `robotiq` keeps all runners working with no edits.

### Branch points (5)

1. **`__init__` gripper setup**
   - `robotiq`: build `rq.RobotiqGripper`, `activate()`, `calibrate(...)`,
     `open()`, set `_gripper_open_threshold_mm` — exactly as now.
   - `franka`: no device. Initialize `self._gripper_byte = self.open_gripper_action`
     (start open).
   - `import pyrobotiqgripper` becomes **lazy**, done inside the robotiq branch,
     so a Franka-only machine without the robotiq library still imports and runs.

2. **`open_gripper` / `close_gripper`**
   - `robotiq`: `self.gripper.open()` / `self.gripper.close()` (blocking), as now.
     `num_steps` ignored.
   - `franka`: set `self._gripper_byte` to the open/close value, then stream
     `[current_joints, byte]` for `num_steps` ticks via `JOINT_POSITION` control
     (exact old code).

3. **Move functions** (`move_to_joints`, `move_along_trajectory`, `osc_move`)
   - Gripper byte used in the action = `GRIPPER_NOOP` when `robotiq` (unchanged),
     else `self._gripper_byte` when `franka`.
   - The `gripper_state` parameter stays ignored (`del gripper_state`) in **both**
     modes. See safety note below.

4. **`get_gripper_state`**
   - `robotiq`: `position_mm()` vs `_gripper_open_threshold_mm` (as now).
   - `franka`: `_gripper_state_buffer[-1].width`,
     `OPEN if abs(width) < 0.01 else CLOSED` (old behavior).

5. **`get_qpos`**
   - `franka`: guard on `_gripper_state_buffer` being populated (old behavior),
     since `get_gripper_state` reads it.
   - `robotiq`: unchanged.

### Safety note — Franka holds the byte from open/close, not from the arg

Old Franka move functions streamed the **passed** `gripper_state` every tick.
The current runners are Robotiq-tuned: they pass `open_gripper_action` (1.0 =
open) into move functions even during a carry, and rely on an explicit
`close_gripper()` to keep holding the object. If Franka mode used the passed
argument, the gripper would open mid-carry and **drop the object**.

Therefore Franka mode tracks `self._gripper_byte`, updated **only** by
`open_gripper()` / `close_gripper()`, and move functions stream that held byte.
Existing runners call open/close at the correct moments, so the held byte is
correct throughout each trajectory. The `gripper_state` parameter remains
ignored in both modes; the public API is unchanged.

### Byte values

- Open byte: `self.open_gripper_action = 1.0`
- Close byte: `self.close_gripper_action = 0.0`
- Robotiq inert byte: `GRIPPER_NOOP = 0.0` (only used in robotiq mode)

## Impact

- `robotiq` mode: zero behavioral change. Same code path, same bytes.
- `franka` mode: opt-in via `FrankaPandaController(gripper_mode="franka")`.
  Restores old streaming gripper, made robust to the current runners by holding
  the last commanded open/close byte through motion.

## Testing

Hardware-dependent (real robot, real grippers) — no unit tests. Manual checks:

1. `robotiq` default: run existing table-bussing flow, confirm unchanged.
2. `franka` mode on a Franka-hand setup: home + `open_gripper()` / `close_gripper()`
   actuate the hand; a pick → carry → place holds the object through the carry
   (byte stays closed until `open_gripper()` at place).
3. Import check: on a machine without `pyrobotiqgripper`, `gripper_mode="franka"`
   imports and constructs without error.
