"""Interactive MuJoCo viewer: replay a recorded G1 teleop episode.

Shows two G1s side by side -- the WBC *action* (``action.wbc``, blue, left) and
the measured *state* (``observation.state``, orange, right) -- as a kinematic
puppet (joint angles written straight into the model; no physics).

Usage:
    python -m teleop_data_analyzer.view_g1_action \
        --dataset-root /path/to/dataset --episode 0

Controls (focus the MuJoCo window):
    space        play / pause
    right / left step one frame (when paused)
    up / down    speed x2 / /2
    o            toggle base orientation (recorded vs. upright)
    r            restart episode
    (plus MuJoCo's own mouse orbit/pan/zoom)
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

if __package__ in (None, ""):
    # Allow running the file directly: `python view_g1_action.py`.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from teleop_data_analyzer.sim.action_data import load_episode
    from teleop_data_analyzer.sim.g1_scene import G1Scene
else:
    from .sim.action_data import load_episode
    from .sim.g1_scene import G1Scene

# GLFW key codes delivered by mujoco.viewer.
_KEY_SPACE = 32
_KEY_RIGHT, _KEY_LEFT, _KEY_UP, _KEY_DOWN = 262, 263, 265, 264
_KEY_O, _KEY_R = ord("O"), ord("R")


def _to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
    """Return a wxyz quaternion (MuJoCo order) from the dataset's stored order."""
    quat = np.asarray(quat, dtype=float)
    if order == "xyzw":
        return np.array([quat[3], quat[0], quat[1], quat[2]])
    return quat  # already wxyz


def _relative_quat(q_t: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Orientation of frame t relative to frame 0, so playback starts upright.

    Avoids guessing the dataset's absolute world frame: we only show how the
    base orientation *changes* over the episode.
    """
    neg0 = np.zeros(4)
    mujoco.mju_negQuat(neg0, q0)        # conjugate of the first-frame quaternion
    rel = np.zeros(4)
    mujoco.mju_mulQuat(rel, q_t, neg0)
    return rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, help="LeRobot dataset root folder")
    ap.add_argument("--episode", type=int, default=0, help="episode index to replay")
    ap.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
    ap.add_argument(
        "--base",
        choices=["upright", "orientation"],
        default="upright",
        help="pin the pelvis upright (default) or apply recorded base orientation",
    )
    ap.add_argument(
        "--quat-order",
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="quaternion order of observation.root_orientation in the dataset",
    )
    ap.add_argument(
        "--hide-ui",
        action="store_true",
        help="hide the MuJoCo side panels (default: panels shown for camera control)",
    )
    args = ap.parse_args()

    ep = load_episode(args.dataset_root, args.episode)
    print(f"Episode {ep.index}: {ep.num_frames} frames @ {ep.fps} fps")
    print(f"Task: {ep.task}")

    scene = G1Scene(ep.joint_names)
    missing = scene.missing_joints
    if missing:
        print(f"[warn] {len(missing)} dataset joints not found in model: {missing}")

    # Precompute base orientation (relative to frame 0) in MuJoCo wxyz order.
    base_quats = None
    if ep.root_orientation.size:
        wxyz = np.array([_to_wxyz(q, args.quat_order) for q in ep.root_orientation])
        q0 = wxyz[0]
        base_quats = np.array([_relative_quat(q, q0) for q in wxyz])

    state = {
        "frame": 0.0,
        "playing": True,
        "speed": args.speed,
        "base": args.base,
    }

    def key_callback(keycode: int) -> None:
        if keycode == _KEY_SPACE:
            state["playing"] = not state["playing"]
        elif keycode == _KEY_RIGHT:
            state["playing"] = False
            state["frame"] = min(ep.num_frames - 1, int(state["frame"]) + 1)
        elif keycode == _KEY_LEFT:
            state["playing"] = False
            state["frame"] = max(0, int(state["frame"]) - 1)
        elif keycode == _KEY_UP:
            state["speed"] = min(16.0, state["speed"] * 2)
        elif keycode == _KEY_DOWN:
            state["speed"] = max(1 / 16, state["speed"] / 2)
        elif keycode == _KEY_O:
            state["base"] = "upright" if state["base"] == "orientation" else "orientation"
        elif keycode == _KEY_R:
            state["frame"] = 0.0

    def pose_current() -> None:
        f = int(state["frame"]) % ep.num_frames
        if state["base"] == "orientation" and base_quats is not None:
            bq = base_quats[f]
        else:
            bq = None
        scene.set_pose(ep.action[f], ep.state[f], action_base_quat=bq, state_base_quat=bq)

    pose_current()
    last = time.time()
    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
        key_callback=key_callback,
        show_left_ui=not args.hide_ui,
        show_right_ui=not args.hide_ui,
    ) as viewer:
        cam = scene.default_camera
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = cam["lookat"]
        viewer.cam.azimuth = cam["azimuth"]
        viewer.cam.elevation = cam["elevation"]
        viewer.cam.distance = cam["distance"]
        while viewer.is_running():
            now = time.time()
            dt = now - last
            last = now
            if state["playing"]:
                state["frame"] += ep.fps * state["speed"] * dt
                if state["frame"] >= ep.num_frames:
                    state["frame"] = 0.0  # loop
            pose_current()
            viewer.sync()
            time.sleep(max(0.0, 1.0 / 120 - (time.time() - now)))


if __name__ == "__main__":
    main()
