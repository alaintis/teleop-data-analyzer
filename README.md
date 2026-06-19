# Tele-op Data Analyzer

Tools to triage and analyze Unitree **G1** teleoperation demonstrations collected
with GR00T **Sonic** whole-body control. See [`plan.md`](plan.md) for the full
design.

This README covers the **G1 action viewer** (the MuJoCo replay). For the
review GUI and batch plotting see `plan.md`.

---

## G1 action viewer

Replays a recorded episode on the G1 in MuJoCo so you can *see what the recorded
action does with the robot*. Two robots are shown side by side:

- **blue (left)** — `action.wbc`, the whole-body-control **target** joint
  positions (the *command*),
- **orange (right)** — `observation.state`, the **measured** joint positions
  (what the robot *actually did*).

Comparing them shows the controller's tracking — where the action and the
realized motion diverge.

### Setup

```bash
./install.sh                       # venv + deps + dataset download
./scripts/fetch_g1_model.sh        # download the G1 model + meshes (~38 MB)
```

`fetch_g1_model.sh` pulls `unitree_g1/g1_with_hands.xml` from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) into
`models/`. That model has the **exact same 43 joint names** as the dataset, so
the replay maps joints purely by name (no hand-built index table).

### Run

```bash
python -m teleop_data_analyzer.view_g1_action \
    --dataset-root data/red_cube_cardbox_all_cleaned_01 \
    --episode 0
```

Options: `--episode N`, `--speed X` (initial playback speed),
`--base {upright,orientation}`, `--quat-order {wxyz,xyzw}`.

### Controls (focus the MuJoCo window)

| Key | Action |
|---|---|
| `space` | play / pause |
| `→` / `←` | step one frame (when paused) |
| `↑` / `↓` | speed ×2 / ÷2 |
| `o` | toggle base orientation (recorded ↔ upright) |
| `r` | restart episode |
| mouse | orbit / pan / zoom (MuJoCo's own controls) |

The MuJoCo side panels are shown by default (use the right panel's camera
controls / mouse to set the view). Pass `--hide-ui` for a clean robots-only
window. To make a tuned view the default, copy the camera numbers from the
panel into `default_camera` in `sim/g1_scene.py`.

---

## ⚠️ Important: what the replay can and cannot show

The viewer is a **kinematic replay** (a puppet): each frame the recorded joint
angles are written straight into the model and forward kinematics is run. There
is **no physics** — no gravity, no contact, no stepping. This is deliberate, and
it has two consequences worth understanding.

**1. Manipulation and articulation are faithful.** The arms, hands (Dex3
fingers), legs, waist and head reproduce the recorded motion exactly, because
every one of the 43 joints is driven directly from the data.

**2. The robot does NOT travel across the floor — and *cannot*, from this data.**
Two separate reasons:

- *No physics to convert stepping into translation.* In a kinematic puppet,
  swinging the leg joints just moves the legs; a real robot moves forward only
  because its feet push against the ground (a contact-physics effect we are not
  simulating). So the legs step **in place**.
- *No world position is recorded (absolute vs. relative).* The dataset stores
  the base **orientation** (`observation.root_orientation`) but **no base
  translation** — there is no world X/Y/Z of the robot anywhere. The Sonic WBC
  is driven in a robot-relative / heading-relative frame
  (`teleop.delta_heading`, `teleop.planner_movement`, `teleop.planner_speed`
  are velocity/heading *commands*, not measured odometry), so the absolute path
  through the room was never logged and is not faithfully recoverable.

Could we instead run a *physics* simulation and let it walk? Not faithfully:
the recorded actions were produced for the real robot (and NVIDIA's sim) with
its specific gains, masses and contacts. Replayed **open-loop** into the
menagerie model the dynamics won't match and there's no feedback to correct
drift — the robot would wobble and fall over within a second or two. So
kinematic replay is the honest choice for *viewing the demonstration*.

**Base orientation** is therefore the one base signal we have. By default the
pelvis is pinned **upright** (cleanest for reading arm/hand motion); pass
`--base orientation` (or press `o`) to apply the recorded orientation as a
**relative** rotation (relative to frame 0, so playback starts upright and you
see how the torso leans over the episode). The quaternion storage order is
assumed `wxyz`; use `--quat-order xyzw` if the lean looks wrong.

---

## Layout

```
teleop_data_analyzer/
├── view_g1_action.py        # the interactive viewer CLI (this tool)
├── sim/
│   ├── g1_scene.py          # builds the dual-G1 MuJoCo scene; name-based posing
│   └── action_data.py       # reads action.wbc / observation.state from parquet
├── dataset.py               # (GUI) video-path loader
└── teleop_data_selector_gui.py
models/unitree_g1/           # G1 model + meshes (downloaded, git-ignored)
scripts/fetch_g1_model.sh    # downloads the model
```
