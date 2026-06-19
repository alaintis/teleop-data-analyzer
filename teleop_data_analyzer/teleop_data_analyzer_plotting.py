"""Batch and per-episode metric plotting for tele-op datasets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
try:
    from teleop_data_analyzer.metrics import (
        action_entropy,
        gripper_aperture,
        jerk,
        joint_velocity_variance,
    )
    from teleop_data_analyzer.sim.action_data import list_episodes, load_episode
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from teleop_data_analyzer.metrics import (
        action_entropy,
        gripper_aperture,
        jerk,
        joint_velocity_variance,
    )
    from teleop_data_analyzer.sim.action_data import list_episodes, load_episode


def _pyplot(save: bool):
    if save:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _episode_indices(dataset_root: str) -> list[int]:
    indices = []
    for offset, path in enumerate(list_episodes(dataset_root)):
        match = re.search(r"episode_(\d+)\.parquet$", Path(path).name)
        indices.append(int(match.group(1)) if match else offset)
    return indices


def _compute_metrics(motion, entropy_window: int, entropy_bins: int) -> dict[str, np.ndarray]:
    left, right = gripper_aperture(motion.state, motion.joint_names)
    return {
        "velocity_variance": joint_velocity_variance(motion.state, motion.fps),
        "jerk": jerk(motion.state, motion.fps),
        "gripper_left": left,
        "gripper_right": right,
        "action_entropy": action_entropy(
            motion.action,
            window=entropy_window,
            bins=entropy_bins,
        ),
    }


def _load_episode_metrics(
    dataset_root: str, episode_index: int, entropy_window: int, entropy_bins: int
) -> dict:
    motion = load_episode(dataset_root, episode_index)
    return {
        "index": motion.index,
        "fps": motion.fps,
        "frames": motion.num_frames,
        "task": motion.task,
        "metrics": _compute_metrics(motion, entropy_window, entropy_bins),
    }


def _series_x(values: np.ndarray, fps: float) -> np.ndarray:
    return np.arange(values.shape[0], dtype=np.float64) / float(fps)


def _style_axis(ax):
    ax.grid(True, color="#ddd", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("time (s)")


def _plot_series(ax, episode: dict, key: str, title: str, color: str):
    values = episode["metrics"][key]
    ax.plot(_series_x(values, episode["fps"]), values, color=color, linewidth=1.4)
    ax.set_title(title)
    _style_axis(ax)


def _plot_gripper_pair(ax, episode: dict):
    left = episode["metrics"]["gripper_left"]
    right = episode["metrics"]["gripper_right"]
    ax.plot(_series_x(left, episode["fps"]), left, color="#1f77b4", linewidth=1.3, label="left")
    ax.plot(
        _series_x(right, episode["fps"]),
        right,
        color="#ff7f0e",
        linewidth=1.3,
        label="right",
    )
    ax.set_title("gripper aperture")
    ax.legend(loc="best")
    _style_axis(ax)


def plot_single_episode(plt, episode: dict, interactive_slider: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), num=f"episode {episode['index']} metrics")
    axes = axes.ravel()
    _plot_series(axes[0], episode, "velocity_variance", "joint velocity variance", "#1f77b4")
    _plot_series(axes[1], episode, "jerk", "jerk", "#d62728")
    _plot_gripper_pair(axes[2], episode)
    _plot_series(axes[3], episode, "action_entropy", "action entropy", "#bc8f00")

    playheads = [
        ax.axvline(0.0, color="#222", linestyle="--", linewidth=1.1)
        for ax in axes
    ]
    title = f"Episode {episode['index']} - {episode['frames']} frames @ {episode['fps']:g} fps"
    if episode["task"]:
        title = f"{title} - {episode['task']}"
    fig.suptitle(title)

    if interactive_slider and episode["frames"] > 1:
        from matplotlib.widgets import Slider

        fig.subplots_adjust(bottom=0.13, top=0.91)
        slider_ax = fig.add_axes([0.16, 0.045, 0.72, 0.03])
        slider = Slider(
            slider_ax,
            "Frame",
            0,
            episode["frames"] - 1,
            valinit=0,
            valstep=1,
        )

        def update(value):
            x = int(value) / episode["fps"]
            for line in playheads:
                line.set_xdata([x, x])
            fig.canvas.draw_idle()

        slider.on_changed(update)
        fig._teleop_frame_slider = slider
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


def _pad_stack(series: list[np.ndarray]) -> np.ndarray:
    max_len = max(values.shape[0] for values in series)
    stack = np.full((len(series), max_len), np.nan, dtype=np.float64)
    for row, values in enumerate(series):
        stack[row, : values.shape[0]] = values
    return stack


def _plot_overlay_metric(ax, episodes: list[dict], key: str, title: str, color: str):
    for episode in episodes:
        values = episode["metrics"][key]
        ax.plot(_series_x(values, episode["fps"]), values, color=color, linewidth=0.7, alpha=0.22)

    stack = _pad_stack([episode["metrics"][key] for episode in episodes])
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    x = np.arange(mean.shape[0], dtype=np.float64) / episodes[0]["fps"]
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2.0, label="mean")
    ax.set_title(title)
    ax.legend(loc="best")
    _style_axis(ax)


def _plot_overlay_grippers(ax, episodes: list[dict]):
    for episode in episodes:
        left = episode["metrics"]["gripper_left"]
        right = episode["metrics"]["gripper_right"]
        ax.plot(_series_x(left, episode["fps"]), left, color="#1f77b4", linewidth=0.6, alpha=0.16)
        ax.plot(_series_x(right, episode["fps"]), right, color="#ff7f0e", linewidth=0.6, alpha=0.16)

    for key, color, label in (
        ("gripper_left", "#1f77b4", "left mean"),
        ("gripper_right", "#ff7f0e", "right mean"),
    ):
        stack = _pad_stack([episode["metrics"][key] for episode in episodes])
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        x = np.arange(mean.shape[0], dtype=np.float64) / episodes[0]["fps"]
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2.0, label=label)

    ax.set_title("gripper aperture")
    ax.legend(loc="best")
    _style_axis(ax)


def plot_all_episodes_overlay(plt, episodes: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), num="metric overlays")
    axes = axes.ravel()
    _plot_overlay_metric(
        axes[0],
        episodes,
        "velocity_variance",
        "joint velocity variance",
        "#1f77b4",
    )
    _plot_overlay_metric(axes[1], episodes, "jerk", "jerk", "#d62728")
    _plot_overlay_grippers(axes[2], episodes)
    _plot_overlay_metric(axes[3], episodes, "action_entropy", "action entropy", "#bc8f00")
    fig.suptitle(f"Metric overlays across {len(episodes)} episodes")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _mean(values: np.ndarray) -> float:
    return float(np.nanmean(values))


def _peak(values: np.ndarray) -> float:
    return float(np.nanmax(values))


def _mean_gripper(metrics: dict[str, np.ndarray]) -> float:
    return float(np.nanmean(np.concatenate([metrics["gripper_left"], metrics["gripper_right"]])))


def _summary_axis(ax, positions: np.ndarray, labels: list[str], values: list[float], title: str, color: str):
    ax.bar(positions, values, color=color, alpha=0.75)
    ax.scatter(positions, values, color="#222", s=12, zorder=3)
    ax.set_title(title)
    ax.grid(True, axis="y", color="#ddd", linewidth=0.6, alpha=0.8)
    step = max(1, len(labels) // 20)
    ax.set_xticks(positions[::step], labels[::step], rotation=90 if len(labels) > 12 else 0)
    ax.set_xlabel("episode")


def plot_episode_summary(plt, episodes: list[dict]):
    labels = [str(episode["index"]) for episode in episodes]
    positions = np.arange(len(episodes))
    summaries = [
        (
            "mean velocity variance",
            [_mean(ep["metrics"]["velocity_variance"]) for ep in episodes],
            "#1f77b4",
        ),
        ("mean jerk", [_mean(ep["metrics"]["jerk"]) for ep in episodes], "#d62728"),
        (
            "mean gripper aperture",
            [_mean_gripper(ep["metrics"]) for ep in episodes],
            "#2ca02c",
        ),
        (
            "peak action entropy",
            [_peak(ep["metrics"]["action_entropy"]) for ep in episodes],
            "#bc8f00",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), num="episode summaries")
    for ax, (title, values, color) in zip(axes.ravel(), summaries, strict=True):
        _summary_axis(ax, positions, labels, values, title, color)
    fig.suptitle("Per-episode metric summaries")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _save_figures(figures: list, save_path: str):
    path = Path(save_path).expanduser()
    if len(figures) == 1:
        if path.suffix:
            outputs = [path]
        else:
            path.mkdir(parents=True, exist_ok=True)
            outputs = [path / "episode_metrics.png"]
    elif path.suffix:
        outputs = [
            path,
            path.with_name(f"{path.stem}_summary{path.suffix}"),
        ]
    else:
        path.mkdir(parents=True, exist_ok=True)
        outputs = [
            path / "metrics_overlay.png",
            path / "metrics_summary.png",
        ]

    for fig, output in zip(figures, outputs, strict=True):
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"saved {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root")
    parser.add_argument("--episode", type=int, help="plot a single episode index")
    parser.add_argument("--save", help="save PNG figure(s) instead of showing a window")
    parser.add_argument("--entropy-window", type=int, default=16, help="entropy window")
    parser.add_argument("--entropy-bins", type=int, default=16, help="entropy histogram bins")
    args = parser.parse_args(argv)

    plt = _pyplot(save=bool(args.save))
    dataset_root = str(Path(args.dataset_root).expanduser())

    if args.episode is not None:
        episode = _load_episode_metrics(
            dataset_root,
            args.episode,
            args.entropy_window,
            args.entropy_bins,
        )
        figures = [plot_single_episode(plt, episode, interactive_slider=not args.save)]
    else:
        episodes = []
        for episode_index in _episode_indices(dataset_root):
            print(f"loading episode {episode_index}")
            episodes.append(
                _load_episode_metrics(
                    dataset_root,
                    episode_index,
                    args.entropy_window,
                    args.entropy_bins,
                )
            )
        figures = [
            plot_all_episodes_overlay(plt, episodes),
            plot_episode_summary(plt, episodes),
        ]

    if args.save:
        _save_figures(figures, args.save)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

