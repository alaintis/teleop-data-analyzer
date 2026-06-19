# Tele-op Data Analyzer — Claude Code Guide

## What this project is

Tools to triage and analyze Unitree **G1** teleoperation demonstrations collected with GR00T **Sonic** whole-body control. Datasets are in [LeRobot format](https://github.com/huggingface/lerobot) (parquet + MP4 videos).

Three entry points:
- **Review GUI** (`teleop_data_selector_gui.py`) — 3 camera feeds + metric plots + synced MuJoCo G1 replay, keyboard-driven episode triage
- **Batch plotter** (`teleop_data_analyzer_plotting.py`) — metric overview across all episodes, no Qt GUI
- **Action viewer** (`view_g1_action.py`) — standalone interactive MuJoCo viewer for a single episode

## Environment setup

```bash
./install.sh                       # creates .venv, installs requirements.txt, downloads sample dataset
./scripts/fetch_g1_model.sh        # downloads G1 MuJoCo model into models/ (~38 MB)
source .venv/bin/activate
```

Always use `.venv/bin/python` (or activate first). The project has no `setup.py`/`pyproject.toml`; run as modules (`python -m teleop_data_analyzer.<module>`).

## Running the tools

```bash
# Review GUI (3 cameras + metrics + MuJoCo pane)
python -m teleop_data_analyzer.teleop_data_selector_gui \
    --dataset-root data/red_cube_cardbox_all_cleaned_01

# Camera + metrics only (skip MuJoCo)
python -m teleop_data_analyzer.teleop_data_selector_gui \
    --dataset-root data/red_cube_cardbox_all_cleaned_01 --no-sim

# Camera feeds only — no metrics, no MuJoCo (fastest, stutter-free; no parquet load)
python -m teleop_data_analyzer.teleop_data_selector_gui \
    --dataset-root data/red_cube_cardbox_all_cleaned_01 --cameras-only

# Batch metric plots across all episodes
python -m teleop_data_analyzer.teleop_data_analyzer_plotting \
    --dataset-root data/red_cube_cardbox_all_cleaned_01

# Single-episode interactive MuJoCo viewer
python -m teleop_data_analyzer.view_g1_action \
    --dataset-root data/red_cube_cardbox_all_cleaned_01 --episode 0
```

## Project layout

```
teleop_data_analyzer/
├── teleop_data_selector_gui.py      # PySide6 review GUI
├── view_g1_action.py                # standalone MuJoCo CLI viewer
├── teleop_data_analyzer_plotting.py # batch metric plots (matplotlib)
├── dataset.py                       # LeRobot dataset loader (video path resolution)
├── metrics.py                       # shared quality metrics (velocity variance, jerk, entropy, gripper aperture)
└── sim/
    ├── g1_scene.py                  # builds dual-G1 MuJoCo XML scene; name-based joint posing
    ├── action_data.py               # reads action.wbc / observation.state from parquet
    └── replay.py                    # offscreen renderer feeding the GUI's MuJoCo pane
models/unitree_g1/                   # G1 model + meshes (git-ignored, fetched by script)
assets/                              # README screenshots
scripts/fetch_g1_model.sh            # model downloader
install.sh                           # venv + deps + dataset downloader
requirements.txt                     # pinned deps: numpy, pandas, pyarrow, matplotlib, PySide6, opencv-python-headless, mujoco, huggingface_hub
```

## Dataset format (LeRobot)

Each dataset root contains:
- `meta/info.json` — fps, chunks_size, video_path template
- `meta/episodes.jsonl` — per-episode task labels
- `videos/chunk-NNN/<camera_key>/episode_NNNNNN.mp4` — MP4 video per camera
- `data/chunk-NNN/episode_NNNNNN.parquet` — per-frame joint data

Key parquet columns:
- `observation.state` — 43 joint positions (measured)
- `action.wbc` — 43 joint position targets (commanded)
- `observation.root_orientation` — pelvis quaternion (wxyz order by default)

G1 joint layout (43 joints): left hand at indices 22–28, right hand at 36–42. The MuJoCo model (`g1_with_hands.xml`) uses the exact same joint names, so posing is done by name lookup.

## Key metrics (`metrics.py`)

| Function | What it measures |
|---|---|
| `joint_velocity_variance(state, fps)` | variance across joint velocities per frame (finite-difference) |
| `jerk(state, fps)` | L2 norm of 3rd derivative per frame |
| `gripper_aperture(state)` | mean left/right hand joint position over time |
| `action_entropy(action, window, bins)` | sliding-window Shannon entropy of action deltas (bits) |

All functions take `(frames, joints)` numpy arrays.

## MuJoCo replay details

The viewer is a **kinematic puppet** — joint angles are written directly each frame, no physics/gravity/contact. Consequence: the robot steps in place (no world translation is recorded in the dataset). The `--base orientation` flag applies the recorded pelvis quaternion as a relative rotation from frame 0. See the README's "Important" section for full explanation.

## Common development tasks

**Add a new metric:** add a function to `metrics.py` following the existing pattern (takes ndarray, returns ndarray), then wire it into `teleop_data_selector_gui.py` and `teleop_data_analyzer_plotting.py`.

**Add a new camera key:** update `EGO_KEY`, `LEFT_WRIST_KEY`, `RIGHT_WRIST_KEY` constants in `dataset.py` and the layout in `teleop_data_selector_gui.py`.

**Change the default MuJoCo camera:** edit `default_camera` and `fovy` in `sim/g1_scene.py`.

**Test without a display:** the batch plotter (`teleop_data_analyzer_plotting.py`) runs headlessly with `--save path/to/output.png`.

## What's git-ignored

- `.venv/` — Python virtual environment
- `models/` — G1 MuJoCo model (downloaded by `fetch_g1_model.sh`)
- `data/` — datasets (downloaded by `install.sh`)
- `__pycache__/`, `*.pyc`
