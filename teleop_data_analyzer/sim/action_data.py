"""Self-contained reader for the per-timestep arrays the G1 action viewer needs.

Kept separate from the GUI's video-oriented ``dataset.py`` so the two efforts
don't step on each other. Reads only the three parquet columns we replay.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

ACTION_KEY = "action.wbc"            # WBC target joint positions  -> "the action"
STATE_KEY = "observation.state"      # measured joint positions    -> "what it did"
ROOT_ORIENT_KEY = "observation.root_orientation"  # base quaternion (orientation only)


@dataclass
class EpisodeMotion:
    index: int
    fps: float
    joint_names: list[str]
    action: np.ndarray            # (T, 43)
    state: np.ndarray             # (T, 43)
    root_orientation: np.ndarray  # (T, 4), quaternion as stored in the dataset
    task: str

    @property
    def num_frames(self) -> int:
        return int(self.action.shape[0])


def _load_tasks(meta_dir: str) -> dict[int, str]:
    tasks: dict[int, str] = {}
    path = os.path.join(meta_dir, "tasks.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    tasks[int(rec["task_index"])] = rec["task"]
    return tasks


def list_episodes(root: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files under {root}/data")
    return paths


def load_episode(root: str, episode_index: int = 0) -> EpisodeMotion:
    """Read one episode's action / state / base-orientation time series."""
    root = os.path.abspath(root)
    meta_dir = os.path.join(root, "meta")
    with open(os.path.join(meta_dir, "info.json")) as f:
        info = json.load(f)
    fps = float(info.get("fps", 50))
    joint_names = info["features"][STATE_KEY]["names"]

    paths = list_episodes(root)
    match = [p for p in paths if f"episode_{episode_index:06d}.parquet" in os.path.basename(p)]
    path = match[0] if match else paths[episode_index]

    table = pq.read_table(path, columns=[ACTION_KEY, STATE_KEY, ROOT_ORIENT_KEY, "task_index"])
    action = np.asarray(table.column(ACTION_KEY).to_pylist(), dtype=np.float64)
    state = np.asarray(table.column(STATE_KEY).to_pylist(), dtype=np.float64)
    root_orient = np.asarray(table.column(ROOT_ORIENT_KEY).to_pylist(), dtype=np.float64)
    task_idx = int(table.column("task_index").to_pylist()[0]) if table.num_rows else 0

    return EpisodeMotion(
        index=episode_index,
        fps=fps,
        joint_names=joint_names,
        action=action,
        state=state,
        root_orientation=root_orient,
        task=_load_tasks(meta_dir).get(task_idx, ""),
    )
