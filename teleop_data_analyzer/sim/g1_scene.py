"""Build a MuJoCo scene that puppets the Unitree G1 from recorded joint data.

This is a *kinematic* replay: each frame we write joint angles straight into
``qpos`` and call ``mj_forward`` (pose only -- no gravity, no contact, no
stepping). The recorded ``action.wbc`` / ``observation.state`` are 43 named
joint positions whose names match ``g1_with_hands.xml`` exactly, so the mapping
is purely by name.

Two robots are placed side by side so the WBC *action* (the command) and the
*state* (what the robot actually did) can be compared at a glance.

Base note: the dataset records base *orientation* but no base *translation*
(see README), so each robot stands at a fixed spot. The pelvis free joint is
swapped for a ball joint -- it carries orientation only -- which is also what
lets the two robots attach into one scene (a free joint must stay top-level).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

_HERE = os.path.dirname(__file__)
DEFAULT_MODEL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "models", "unitree_g1", "g1_with_hands.xml")
)

# Name of the pelvis free joint in g1_with_hands.xml.
_BASE_JOINT = "floating_base_joint"

ACTION_COLOR = (0.20, 0.55, 1.00, 1.0)   # blue  -> the commanded action
STATE_COLOR = (1.00, 0.55, 0.20, 1.0)    # orange -> what the robot actually did


def _make_child(model_path: str) -> mujoco.MjSpec:
    """Load g1_with_hands and convert its pelvis free joint to a ball joint."""
    spec = mujoco.MjSpec.from_file(model_path)
    # Keyframes encode a qpos of the original (free-joint) size; drop them so the
    # model still compiles after we shrink the base joint.
    for key in list(spec.keys):
        spec.delete(key)
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            joint.type = mujoco.mjtJoint.mjJNT_BALL
    return spec


@dataclass
class _RobotMap:
    """Where each named joint and the base orientation live in qpos for one robot."""

    qpos_adr: np.ndarray   # (43,) qpos address per dataset joint, -1 if absent
    base_qpos_adr: int     # start of the ball-joint quaternion (wxyz), or -1


class G1Scene:
    """A two-robot G1 scene with name-based kinematic posing."""

    def __init__(
        self,
        joint_names: list[str],
        model_path: str = DEFAULT_MODEL,
        spacing: float = 0.9,
        base_height: float = 0.0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"G1 model not found at {model_path}.\n"
                "Run scripts/fetch_g1_model.sh to download it."
            )
        self.joint_names = joint_names

        parent = mujoco.MjSpec()
        parent.modelname = "g1_action_vs_state"
        parent.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        parent.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[6, 6, 0.1],
            rgba=[0.25, 0.26, 0.28, 1.0],
        )

        # prefix -> x offset; action on the left, state on the right.
        self._layout = {
            "act_": -spacing / 2.0,
            "state_": spacing / 2.0,
        }
        for prefix, dx in self._layout.items():
            child = _make_child(model_path)
            frame = parent.worldbody.add_frame(pos=[dx, 0, base_height])
            frame.attach_body(child.worldbody, prefix, "")

        self.model = parent.compile()
        self.data = mujoco.MjData(self.model)

        self._maps = {p: self._build_map(p) for p in self._layout}
        self._tint("act_", ACTION_COLOR)
        self._tint("state_", STATE_COLOR)

        # Field of view (degrees) for the free camera.
        self.model.vis.global_.fovy = 42.0
        # Free-camera pose the viewer should open with (frames both robots).
        self.default_camera = {
            "lookat": (0.0, 0.0, 0.8),
            "azimuth": 140.0,
            "elevation": -8.0,
            "distance": 3.0,
        }

        # Neutral standing pose, base upright.
        mujoco.mj_forward(self.model, self.data)

    # -- setup helpers -------------------------------------------------

    def _build_map(self, prefix: str) -> _RobotMap:
        m = self.model
        adr = np.full(len(self.joint_names), -1, dtype=int)
        for i, name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
            if jid >= 0:
                adr[i] = m.jnt_qposadr[jid]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + _BASE_JOINT)
        base_adr = int(m.jnt_qposadr[bid]) if bid >= 0 else -1
        return _RobotMap(qpos_adr=adr, base_qpos_adr=base_adr)

    def _tint(self, prefix: str, rgba) -> None:
        """Flat-colour every geom of one robot so action/state are distinguishable.

        Most G1 visual geoms are unnamed, so identify a geom's robot by the name
        of the body it belongs to (bodies carry the attach prefix).
        """
        m = self.model
        for gid in range(m.ngeom):
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
            if body_name.startswith(prefix):
                m.geom_matid[gid] = -1          # ignore the menagerie material
                m.geom_rgba[gid] = rgba

    @property
    def missing_joints(self) -> list[str]:
        adr = self._maps["act_"].qpos_adr
        return [n for n, a in zip(self.joint_names, adr) if a < 0]

    # -- posing --------------------------------------------------------

    def set_pose(
        self,
        action_q: np.ndarray,
        state_q: np.ndarray | None = None,
        action_base_quat: np.ndarray | None = None,
        state_base_quat: np.ndarray | None = None,
    ) -> None:
        """Pose the two robots for one frame and run forward kinematics.

        ``*_q`` are length-43 joint vectors in dataset joint order.
        ``*_base_quat`` are wxyz quaternions for the pelvis; ``None`` = upright.
        """
        self._apply("act_", action_q, action_base_quat)
        if state_q is not None:
            self._apply("state_", state_q, state_base_quat)
        mujoco.mj_forward(self.model, self.data)

    def _apply(self, prefix: str, q: np.ndarray, base_quat: np.ndarray | None) -> None:
        rmap = self._maps[prefix]
        qpos = self.data.qpos
        adr = rmap.qpos_adr
        valid = adr >= 0
        qpos[adr[valid]] = np.asarray(q)[valid]
        if rmap.base_qpos_adr >= 0:
            b = rmap.base_qpos_adr
            if base_quat is None:
                qpos[b : b + 4] = (1.0, 0.0, 0.0, 0.0)
            else:
                quat = np.asarray(base_quat, dtype=float)
                n = np.linalg.norm(quat)
                qpos[b : b + 4] = quat / n if n > 0 else (1.0, 0.0, 0.0, 0.0)
