"""Quality metrics shared by the GUI and batch plotting tools."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

LEFT_HAND_SLICE = slice(22, 29)
RIGHT_HAND_SLICE = slice(36, 43)


def _as_position_array(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array shaped (frames, joints)")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one joint column")
    return arr


def _check_fps(fps: float) -> float:
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    return fps


def _time_gradient(values: np.ndarray, fps: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values)
    edge_order = 2 if values.shape[0] >= 3 else 1
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=edge_order)


def joint_velocity_variance(state: np.ndarray, fps: float) -> np.ndarray:
    """Variance across joint velocities for each frame.

    The dataset currently stores joint positions but not recorded velocities, so
    velocity is estimated with a time-aligned finite difference.
    """

    state = _as_position_array("state", state)
    fps = _check_fps(fps)
    velocity = _time_gradient(state, fps)
    return np.var(velocity, axis=1)


def jerk(state: np.ndarray, fps: float) -> np.ndarray:
    """L2 norm of the third finite-difference derivative per frame."""

    values = _as_position_array("state", state)
    fps = _check_fps(fps)
    for _ in range(3):
        values = _time_gradient(values, fps)
    return np.linalg.norm(values, axis=1)


def gripper_aperture(
    state: np.ndarray, joint_names: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean left/right hand joint position over time.

    The verified G1 dataset has 43 joint positions with left hand joints at
    indices 22-28 and right hand joints at indices 36-42.
    """

    state = _as_position_array("state", state)
    if state.shape[1] < RIGHT_HAND_SLICE.stop:
        detail = ""
        if joint_names is not None:
            detail = f" ({len(joint_names)} joint names provided)"
        raise ValueError(
            "state must contain at least 43 joint columns for G1 hand aperture"
            f"{detail}"
        )
    left = np.mean(state[:, LEFT_HAND_SLICE], axis=1)
    right = np.mean(state[:, RIGHT_HAND_SLICE], axis=1)
    return left, right


def action_entropy(action: np.ndarray, window: int = 16, bins: int = 16) -> np.ndarray:
    """Sliding-window Shannon entropy of flattened action deltas.

    Entropy is measured in bits. The first frame is aligned with a zero delta so
    the returned series has the same length as ``action``.
    """

    action = _as_position_array("action", action)
    window = int(window)
    bins = int(bins)
    if window <= 0:
        raise ValueError("window must be positive")
    if bins <= 0:
        raise ValueError("bins must be positive")

    deltas = np.diff(action, axis=0, prepend=action[:1])
    entropy = np.zeros(action.shape[0], dtype=np.float64)
    for frame in range(action.shape[0]):
        start = max(0, frame - window + 1)
        values = deltas[start : frame + 1].ravel()
        if values.size == 0 or np.all(values == values[0]):
            continue
        counts, _ = np.histogram(values, bins=bins)
        counts = counts[counts > 0]
        if counts.size == 0:
            continue
        probs = counts / counts.sum()
        entropy[frame] = -float(np.sum(probs * np.log2(probs)))
    return entropy

