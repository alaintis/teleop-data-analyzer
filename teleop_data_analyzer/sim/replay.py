"""Offscreen G1 replay controller for embedding the sim inside a Qt GUI.

Wraps :class:`~teleop_data_analyzer.sim.g1_scene.G1Scene` with a
``mujoco.Renderer`` so a single frame can be posed and rendered to an RGB
ndarray on demand. Unlike :mod:`view_g1_action` (which owns its own GLFW
window and playback loop), this class renders into an image the host GUI
drives — letting the MuJoCo pane follow the camera viewer's episode / frame.

The model and renderer are built lazily on the first ``load_episode`` (the
joint names come from the dataset), and reused across episodes since the
joint layout is constant.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .action_data import load_episode
from .g1_scene import G1Scene


def to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
    """Return a wxyz quaternion (MuJoCo order) from the dataset's stored order."""
    quat = np.asarray(quat, dtype=float)
    if order == "xyzw":
        return np.array([quat[3], quat[0], quat[1], quat[2]])
    return quat  # already wxyz


def relative_quat(q_t: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Orientation of frame t relative to frame 0, so playback starts upright.

    Avoids guessing the dataset's absolute world frame: we only show how the
    base orientation *changes* over the episode.
    """
    neg0 = np.zeros(4)
    mujoco.mju_negQuat(neg0, q0)        # conjugate of the first-frame quaternion
    rel = np.zeros(4)
    mujoco.mju_mulQuat(rel, q_t, neg0)
    return rel


class G1Replay:
    """Pose-and-render one episode at a time into RGB frames.

    Parameters
    ----------
    dataset_root:
        LeRobot dataset root (same root the camera viewer reads).
    width, height:
        Offscreen render resolution. The host pane scales the result to fit.
    apply_orientation:
        If True, apply the recorded base orientation (relative to frame 0);
        otherwise pin the pelvis upright.
    quat_order:
        Quaternion order of ``observation.root_orientation`` in the dataset.
    """

    def __init__(
        self,
        dataset_root,
        width: int = 640,
        height: int = 480,
        apply_orientation: bool = False,
        quat_order: str = "wxyz",
    ):
        self.root = str(dataset_root)
        self.width = int(width)
        self.height = int(height)
        self.apply_orientation = apply_orientation
        self.quat_order = quat_order

        self.scene: G1Scene | None = None
        self.renderer: mujoco.Renderer | None = None
        self.cam: mujoco.MjvCamera | None = None
        self.motion = None
        self._base_quats: np.ndarray | None = None

    # -- setup ---------------------------------------------------------------
    def _ensure_scene(self, joint_names: list[str]) -> None:
        if self.scene is not None:
            return
        self.scene = G1Scene(joint_names)
        self.renderer = mujoco.Renderer(self.scene.model, self.height, self.width)
        cam = mujoco.MjvCamera()
        cfg = self.scene.default_camera
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = cfg["lookat"]
        cam.azimuth = cfg["azimuth"]
        cam.elevation = cfg["elevation"]
        cam.distance = cfg["distance"]
        self.cam = cam

    # -- per-episode ---------------------------------------------------------
    def load_episode(self, episode_index: int) -> None:
        """Load the action/state time series for ``episode_index`` and prep it."""
        self.motion = load_episode(self.root, episode_index)
        self._ensure_scene(self.motion.joint_names)

        self._base_quats = None
        if self.apply_orientation and self.motion.root_orientation.size:
            wxyz = np.array(
                [to_wxyz(q, self.quat_order) for q in self.motion.root_orientation]
            )
            q0 = wxyz[0]
            self._base_quats = np.array([relative_quat(q, q0) for q in wxyz])

    @property
    def num_frames(self) -> int:
        return self.motion.num_frames if self.motion is not None else 0

    # -- rendering -----------------------------------------------------------
    def render(self, frame_idx: int) -> np.ndarray:
        """Pose both robots at ``frame_idx`` and return an HxWx3 RGB uint8 image."""
        if self.motion is None or self.scene is None or self.renderer is None:
            raise RuntimeError("load_episode() must be called before render()")
        m = self.motion
        f = max(0, min(m.num_frames - 1, int(frame_idx)))
        bq = self._base_quats[f] if self._base_quats is not None else None
        self.scene.set_pose(m.action[f], m.state[f], action_base_quat=bq, state_base_quat=bq)
        self.renderer.update_scene(self.scene.data, camera=self.cam)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
