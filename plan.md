# Tele-op Data Analyzer — Project Plan

## 1. Goal

Two Python tools to **triage** and **analyze** teleoperation demonstrations
collected on the Unitree **G1** humanoid running GR00T **Sonic** whole-body
control ([GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)):

1. **`teleop_data_selector_gui.py`** — an interactive per-sample review GUI to
   watch each demonstration, inspect quality metrics, optionally fix swapped
   wrist cameras, and sort each sample into `good/` or `discard/`.
2. **`teleop_data_analyzer_plotting.py`** — a batch script that computes and
   plots the four quality metrics across **all** samples at once, for
   dataset-level overview and outlier spotting.

Both live in a `teleop_data_analyzer/` Python package and **share** the data
loader and metric implementations (single source of truth).

---

## 2. Input data format

The data is a **LeRobot dataset** exported by the GR00T data collector. Verified
schema (from the GR00T data-collection docs — to be re-confirmed against a real
sample, see §9):

```
<dataset_root>/
├── data/
│   ├── train-00000.parquet          # tabular state/action, possibly many episodes per file
│   └── ...
├── videos/
│   ├── observation.images.ego_view/     episode_000000.mp4 ...   # head/main camera, 480×640×3
│   ├── observation.images.left_wrist/   episode_000000.mp4 ...   # optional (recorded with --record-wrist-cameras)
│   └── observation.images.right_wrist/  episode_000000.mp4 ...
└── meta/
    ├── info.json        # fps, features, shapes, sizes
    ├── modality.json    # GR00T modality config → joint names / index ranges
    ├── episodes.jsonl   # per-episode metadata (length, index, etc.)
    └── tasks.jsonl      # task-prompt definitions
```

**Parquet columns (per timestep):**

| Column | Shape | Meaning |
|---|---|---|
| `observation.state.joint_position`  | `(N,)` | actuated joint positions (rad) |
| `observation.state.joint_velocity`  | `(N,)` | actuated joint velocities (rad/s) |
| `observation.state.body_rotation_6d` | `(6,)` | base orientation (6D rotation) |
| `observation.state.projected_gravity` | `(3,)` | gravity vector in body frame |
| `action.joint_position`             | `(N,)` | teleop target joint positions |
| `action.body_rotation_6d`           | `(6,)` | teleop target body rotation |
| `annotation.human.action.task_description` | str | task prompt |

**Important data gaps (drive decisions below):**
- **No gripper force / torque is recorded** — only joint position & velocity.
- **No explicit base translation** — orientation is available
  (`body_rotation_6d`, `projected_gravity`) but a full floating-base world
  position may not be. Affects the simulator pane (§5.4 / §9).
- `N` (actuated joint count) and which indices are the **gripper/hand** joints
  are **not** in the docs — read them from `meta/modality.json` at runtime.

---

## 3. Confirmed design decisions

| Topic | Decision |
|---|---|
| **GUI framework** | **PySide6 / Qt** — native window; cleanly embeds video frames, Matplotlib `FigureCanvas`, an offscreen MuJoCo render, and global single-key capture. |
| **"Gripper force profile" plot** | No force in data → plot **gripper aperture** = the gripper/hand **joint position(s)** over time as the grip proxy. |
| **"Action entropy" metric** | **Windowed histogram (Shannon) entropy** of per-step action deltas over a sliding window. |
| **`s` (swap wrist cameras)** | Toggles on-screen view **and persists the correction on export** — a sample sent to `good/` has its left/right wrist references physically swapped so the saved sample is corrected. |

---

## 4. Metric definitions (shared `metrics.py`, used by both tools)

`fps` comes from `meta/info.json`; `dt = 1/fps`.

1. **Joint velocity variance** — per timestep, variance across the actuated
   joints of `observation.state.joint_velocity` → a scalar time series.
   *(Default; alternative = rolling per-joint variance — see §9.)*
2. **Jerk (d³pos/dt³)** — third time derivative of `observation.state.joint_position`
   via finite differences, aggregated as the L2 norm across joints per timestep.
3. **Gripper aperture profile** — extract the gripper/hand joint index range
   from `modality.json`; plot those positions (rad) over time.
4. **Action entropy per timestep** — sliding window `W` over the per-step deltas
   of `action.joint_position`; per window, bin the values into a histogram and
   take its Shannon entropy. High = jittery/exploratory, low = smooth.
   Defaults (`W`, bin count) configurable — see §9.

All four return time-aligned 1-D arrays so the GUI can draw a synchronized
playhead and the batch tool can overlay/aggregate them.

---

## 5. Tool 1 — `teleop_data_selector_gui.py` (PySide6)

### 5.1 Layout — 6 panes (2 rows × 3 columns)

```
┌───────────────┬───────────────┬───────────────┐
│  left_wrist   │   ego_view    │  right_wrist   │   top row: camera feeds
│   (camera)    │  (main cam)   │   (camera)     │
├───────────────┼───────────────┼───────────────┤
│  4 metric     │   MuJoCo G1    │  info /        │   bottom row: plots + sim + controls
│  plots (2×2)  │   sim replay   │  controls      │
└───────────────┴───────────────┴───────────────┘
```

- Top row matches the brief: **main (ego) camera in the middle, wrist cameras
  left & right**.
- Bottom-left cell holds the **four metric plots** as a 2×2 Matplotlib figure,
  each with a vertical playhead at the current frame.
- Bottom-middle cell holds the **MuJoCo G1 replay** (Phase 3).
- Bottom-right cell is an **info/controls** pane: sample name, task description,
  frame slider, good/discard counters, and the key legend. *(This 6th pane
  fleshes out "6 parts"; layout is easily adjusted if you'd prefer the four
  plots spread across two bottom cells instead.)*

### 5.2 Playback
Synchronized play/pause across all three videos, the plot playheads, and the
sim. Scrubber/slider, configurable speed. Video decode via PyAV (`av`) or
OpenCV.

### 5.3 Keyboard shortcuts
| Key | Action |
|---|---|
| `s` | Swap left/right wrist view (and mark for persistence on export) |
| `d` | Send current sample → `discard/`, advance to next |
| `g` | Send current sample → `good/`, advance to next |
| `space` | Play / pause |
| `←/→` | Step / scrub frames |
| `n` / `p` | Next / previous sample without sorting |
| `q` | Quit |

### 5.4 Sorting / export (`d` and `g`)
A "sample" = one episode's parquet rows + its three per-camera `episode_XXXXXX.mp4`
files + its metadata entries. The **exact move mechanics depend on the on-disk
layout** (see §9):
- **If one parquet + per-episode files per episode** → move the files into
  `good/` or `discard/`.
- **If episodes are aggregated** into shared parquet/metadata → "moving" means
  copying that episode's rows + videos into a new `good/` or `discard/` LeRobot
  sub-dataset (and optionally rewriting `meta/*` counts), leaving the source
  intact. **Default to non-destructive copy** unless you confirm otherwise.

On export to `good/` with the wrist swap active, the left/right wrist video
references are written swapped so the corrected sample is the one saved.

---

## 6. Tool 2 — `teleop_data_analyzer_plotting.py`

- Iterate every episode in the dataset, compute the four metrics via
  `metrics.py`.
- Produce dataset-level figures: per-metric overlay of all episodes (with
  mean ± band) plus a per-episode summary (e.g. mean/peak jerk, mean entropy)
  to surface outliers.
- CLI: `--dataset-root`, `--out`, metric window/bin params; save PNG(s) and/or
  show interactively.

---

## 7. Package architecture

```
teleop_data_analyzer/
├── __init__.py
├── dataset.py          # LeRobot loader: list episodes, read parquet rows,
│                       #   open the 3 videos, parse meta/ (fps, joint names, gripper idx)
├── metrics.py          # the 4 metric functions (shared)
├── sim/
│   └── g1_mujoco.py    # offscreen MuJoCo renderer + parquet→MJCF joint mapping
├── teleop_data_selector_gui.py        # Tool 1 (PySide6)
└── teleop_data_analyzer_plotting.py   # Tool 2
```

---

## 8. Phasing (incremental; risky sim pane last)

- **Phase 0 — Ground truth.** Point the loader at a real sample; verify layout,
  `N`, joint names, gripper indices, whether wrist cams exist, and the
  aggregated-vs-per-episode question (§9).
- **Phase 1 — `dataset.py` + `metrics.py` + Tool 2.** Batch plotting validates
  the metrics independently of any GUI.
- **Phase 2 — GUI core.** PySide6 window, 3 cameras + 4 plots + synchronized
  playback + `s`/`d`/`g` sorting (no sim yet).
- **Phase 3 — MuJoCo pane.** Offscreen G1 render with verified joint mapping,
  synced to playback.
- **Phase 4 — Polish.** Swap-persist export, counters/undo, config file, packaging.

---

## 9. Assumptions & open questions (please confirm / will verify on real data)

1. **Dataset path** — need a path to one real teleop dataset folder to ground
   Phase 0. *(Searched the machine; none found yet.)*
2. **On-disk layout** — aggregated episodes vs one-file-per-episode. Determines
   whether `d`/`g` move files or copy rows into a sub-dataset. **Defaulting to
   non-destructive copy** until confirmed.
3. **Joint count `N` & gripper indices** — to be read from `modality.json`;
   need to confirm the gripper/hand joint index range for the aperture plot.
4. **MuJoCo G1 model source** — repo `models/` (URDF/XML) vs
   `mujoco_menagerie` `unitree_g1`. Need the MJCF + a verified joint-order
   mapping from the parquet `joint_position` order to MJCF joints.
5. **Simulator fidelity** — with orientation but possibly no base translation,
   the replay may be joints-on-a-fixed/oriented base rather than full
   floating-base motion. Acceptable? (Can revisit if root state turns out to be
   recoverable.)
6. **Joint velocity variance** — across-joints-per-timestep (default) vs
   rolling-window-per-joint.
7. **Action entropy params** — default window `W` (e.g. 16 frames) and histogram
   bin count; confirm or tune.
8. **Python / deps** — Python ≥3.10; PySide6, numpy, pandas + pyarrow,
   matplotlib, av (or opencv-python), mujoco. (lerobot optional, for reuse.)

---

## Appendix — original brief (verbatim)

> Fetch teleop video from folder. The teleop video data comes from the G1 with
> Sonic WBC. Provide a GUI with the main view camera in the middle and the wrist
> camera on the left and right side. `s` swaps left/right wrist (some are
> accidentally swapped); `d` puts the whole sample (video, meta, data) in a
> `discard` folder; `g` puts it in a `good` folder. Below the camera view, plots
> for: Joint velocity variance; Jerk (d³pos/dt³); Gripper force profile; Action
> entropy per timestep. Plus a view of the robot doing the teleop action in
> MuJoCo or another simulator. Screen divided into 6 parts: top row camera feed,
> bottom row plots + simulator view. File: `teleop_data_selector_gui.py` in the
> `teleop_data_analyzer` folder. Then a script `teleop_data_analyzer_plotting.py`
> plotting the same four metrics for all teleop data at once.
