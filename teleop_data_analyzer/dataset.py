"""Minimal LeRobot dataset loader for the tele-op selector GUI.

Phase 2 scope: just enough to enumerate episodes and resolve the three
per-episode camera video paths. Parquet / metric loading comes later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Camera video keys, in the on-screen order we care about.
EGO_KEY = "observation.images.ego_view"
LEFT_WRIST_KEY = "observation.images.left_wrist"
RIGHT_WRIST_KEY = "observation.images.right_wrist"


@dataclass
class Episode:
    index: int
    ego: Path
    left_wrist: Path
    right_wrist: Path
    task: str = ""

    @property
    def name(self) -> str:
        return f"episode_{self.index:06d}"


class Dataset:
    """Reads meta/info.json and resolves video paths for each episode."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing meta/info.json under {self.root}")
        self.info = json.loads(info_path.read_text())

        self.fps: float = float(self.info.get("fps", 30))
        self.chunks_size: int = int(self.info.get("chunks_size", 1000))
        self.video_path_tpl: str = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )

        self._tasks_by_episode = self._load_episode_tasks()
        self.episodes: list[Episode] = self._discover_episodes()
        if not self.episodes:
            raise RuntimeError(f"No episodes with all three camera videos under {self.root}")

    def _load_episode_tasks(self) -> dict[int, str]:
        tasks: dict[int, str] = {}
        ep_jsonl = self.root / "meta" / "episodes.jsonl"
        if ep_jsonl.is_file():
            for line in ep_jsonl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                names = rec.get("tasks") or []
                tasks[int(rec["episode_index"])] = names[0] if names else ""
        return tasks

    def _video_path(self, episode_index: int, video_key: str) -> Path:
        chunk = episode_index // self.chunks_size
        rel = self.video_path_tpl.format(
            episode_chunk=chunk, video_key=video_key, episode_index=episode_index
        )
        return self.root / rel

    def _discover_episodes(self) -> list[Episode]:
        episodes: list[Episode] = []
        ego_dir = self.root / "videos" / "chunk-000" / EGO_KEY
        if not ego_dir.is_dir():
            return episodes
        for ego_file in sorted(ego_dir.glob("episode_*.mp4")):
            try:
                idx = int(ego_file.stem.split("_")[-1])
            except ValueError:
                continue
            left = self._video_path(idx, LEFT_WRIST_KEY)
            right = self._video_path(idx, RIGHT_WRIST_KEY)
            if not (left.is_file() and right.is_file()):
                continue
            episodes.append(
                Episode(
                    index=idx,
                    ego=ego_file,
                    left_wrist=left,
                    right_wrist=right,
                    task=self._tasks_by_episode.get(idx, ""),
                )
            )
        return episodes
