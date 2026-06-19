"""Tele-op data selector GUI — three-camera viewer + MuJoCo G1 replay.

Top row streams the three per-episode camera feeds (left wrist | ego/main |
right wrist). The bottom-right pane shows the MuJoCo G1 kinematic replay
(commanded action vs. measured state) rendered offscreen and kept in sync with
the camera playhead — it follows the same episode and frame, so navigating to
another sample re-loads its motion automatically. (The two remaining bottom
cells are placeholders for the metric plots / info pane.)

The feeds are synchronized and provide the review keyboard shortcuts:

    space   play / pause
    < / >   step one frame (when paused)
    s       swap the two wrist views (and remember it for this sample)
    g       mark current sample "good", advance to next
    d       mark current sample "discard", advance to next
    n / p   next / previous sample (no decision)
    q       quit

`g` / `d` decisions are recorded non-destructively to `decisions.json` in the
dataset root (no files are moved yet); actual export is a later phase.

Usage (either works):
    python teleop_data_selector_gui.py --dataset-root <path>
    python -m teleop_data_analyzer.teleop_data_selector_gui --dataset-root <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Pick a working Qt platform plugin before Qt is imported. On a Wayland session
# the default "xcb" (X11) plugin needs libxcb-cursor0, which may be missing and
# aborts startup. If the user is under Wayland and hasn't forced a platform,
# default to the wayland plugin so the app starts without extra system packages.
if not os.environ.get("QT_QPA_PLATFORM") and os.environ.get("WAYLAND_DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "wayland"

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

# Work whether launched as a module (`python -m ...`) or as a plain script
# (`python teleop_data_selector_gui.py`). In the latter case the package's
# parent dir isn't on sys.path, so add it before importing the package.
try:
    from teleop_data_analyzer.dataset import Dataset, Episode
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from teleop_data_analyzer.dataset import Dataset, Episode

# MuJoCo is optional: if it (or the G1 model) is unavailable the GUI still runs
# as a pure camera viewer and the sim pane shows why it's empty.
try:
    from teleop_data_analyzer.sim.replay import G1Replay
except Exception:  # pragma: no cover - import-time env/dep issues
    G1Replay = None


class VideoSource:
    """Wraps one cv2.VideoCapture: sequential reads for playback, seek for steps."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame = self._read()  # current decoded frame (BGR ndarray or None)

    def _read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def advance(self) -> bool:
        """Read the next frame. Returns False at end-of-stream (frame unchanged)."""
        nxt = self._read()
        if nxt is None:
            return False
        self.frame = nxt
        return True

    def seek(self, idx: int):
        idx = max(0, idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        nxt = self._read()
        if nxt is not None:
            self.frame = nxt

    def restart(self):
        self.seek(0)

    def release(self):
        self.cap.release()


class CameraPane(QtWidgets.QWidget):
    """A titled video pane that scales its frame to fit while keeping aspect."""

    def __init__(self, title: str):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #ddd; font-weight: bold; padding: 2px;")
        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(160, 120)
        self.view.setStyleSheet("background: #111;")
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        layout.addWidget(self.title_label)
        layout.addWidget(self.view, 1)
        self._pixmap: QtGui.QPixmap | None = None

    def set_title(self, title: str):
        self.title_label.setText(title)

    def set_frame(self, frame_bgr):
        if frame_bgr is None:
            self._pixmap = None
            self.view.clear()
            return
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(img.copy())
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        self.view.setPixmap(
            self._pixmap.scaled(
                self.view.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class RenderPane(CameraPane):
    """A titled pane that displays an RGB ndarray (e.g. the MuJoCo render).

    Reuses :class:`CameraPane`'s aspect-preserving scaling but, unlike it,
    takes already-RGB frames and can fall back to a centered status message
    (e.g. when the sim is unavailable).
    """

    def set_rgb(self, rgb):
        if rgb is None:
            self._pixmap = None
            self.view.clear()
            return
        h, w = rgb.shape[:2]
        # QImage needs contiguous memory; render output already is, but copy to
        # be safe and to detach from the renderer's reused buffer.
        img = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(img)
        self._rescale()

    def set_message(self, text: str):
        self._pixmap = None
        self.view.setText(text)
        self.view.setStyleSheet("background: #111; color: #888; padding: 8px;")


class SelectorWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        dataset: Dataset,
        sim_base: str = "upright",
        quat_order: str = "wxyz",
        enable_sim: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.episodes: list[Episode] = dataset.episodes
        self.idx = 0  # index into self.episodes
        self.swap = False
        self.playing = True
        self.frame_pos = 0

        self.decisions_path = dataset.root / "decisions.json"
        self.decisions = self._load_decisions()

        self.left_src: VideoSource | None = None
        self.ego_src: VideoSource | None = None
        self.right_src: VideoSource | None = None

        # MuJoCo replay that follows the selected episode/frame (Phase 3).
        self.sim = None
        self.sim_loaded = False
        self.sim_error: str | None = None
        if not enable_sim:
            self.sim_error = "sim disabled (--no-sim)"
        elif G1Replay is None:
            self.sim_error = "MuJoCo unavailable\n(install `mujoco`)"
        else:
            try:
                self.sim = G1Replay(
                    dataset.root,
                    apply_orientation=(sim_base == "orientation"),
                    quat_order=quat_order,
                )
            except Exception as exc:  # GL context / model missing, etc.
                self.sim_error = f"Sim disabled:\n{exc}"

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        interval = max(1, int(round(1000.0 / dataset.fps)))
        self.timer.setInterval(interval)

        self._load_current()
        self.timer.start()

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Tele-op Data Selector — camera viewer + G1 sim")
        self.resize(1280, 960)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # Top row: the three camera feeds.
        row = QtWidgets.QHBoxLayout()
        self.pane_left = CameraPane("left wrist")
        self.pane_ego = CameraPane("ego (main)")
        self.pane_right = CameraPane("right wrist")
        for p in (self.pane_left, self.pane_ego, self.pane_right):
            row.addWidget(p, 1)
        outer.addLayout(row, 1)

        # Bottom row: [ metric plots | (reserved) | MuJoCo sim ]. The first two
        # cells are placeholders until the metric/info panes land; the third is
        # the MuJoCo G1 replay, synced to the camera playhead.
        bottom = QtWidgets.QHBoxLayout()
        self.pane_plots = RenderPane("metric plots")
        self.pane_plots.set_message("metric plots\n(coming soon)")
        self.pane_info = RenderPane("info / controls")
        self.pane_info.set_message("info / controls\n(coming soon)")
        self.pane_sim = RenderPane("MuJoCo G1 (action vs state)")
        if self.sim_error:
            self.pane_sim.set_message(self.sim_error)
        for p in (self.pane_plots, self.pane_info, self.pane_sim):
            bottom.addWidget(p, 1)
        outer.addLayout(bottom, 1)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color: #ccc; padding: 4px;")
        outer.addWidget(self.status)

        legend = QtWidgets.QLabel(
            "space play/pause   ‹/› step   s swap wrists   "
            "g good   d discard   n/↓ next   p/↑ prev   q quit"
        )
        legend.setStyleSheet(
            "color: #2ecc40; font-size: 18px; font-weight: bold; padding: 4px;"
        )
        outer.addWidget(legend)

        central.setStyleSheet("background: #1b1b1b;")

        # Big episode-number badge overlaid in the top-right corner.
        self.episode_badge = QtWidgets.QLabel(central)
        self.episode_badge.setStyleSheet(
            "color: #fff; font-size: 44px; font-weight: bold; "
            "background: rgba(0, 0, 0, 150); padding: 4px 14px; border-radius: 8px;"
        )
        self.episode_badge.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def _position_badge(self):
        central = self.centralWidget()
        if central is None:
            return
        self.episode_badge.adjustSize()
        margin = 14
        self.episode_badge.move(
            central.width() - self.episode_badge.width() - margin, margin
        )
        self.episode_badge.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_badge()

    # ---- decisions persistence --------------------------------------------
    def _load_decisions(self) -> dict[str, dict]:
        if self.decisions_path.is_file():
            try:
                return json.loads(self.decisions_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_decisions(self):
        try:
            self.decisions_path.write_text(json.dumps(self.decisions, indent=2))
        except OSError as exc:  # pragma: no cover - surfaced in status bar
            self.status.setText(f"Failed to save decisions: {exc}")

    # ---- episode loading ---------------------------------------------------
    def _release_sources(self):
        for src in (self.left_src, self.ego_src, self.right_src):
            if src is not None:
                src.release()
        self.left_src = self.ego_src = self.right_src = None

    def _load_current(self):
        self._release_sources()
        ep = self.episodes[self.idx]
        self.left_src = VideoSource(ep.left_wrist)
        self.ego_src = VideoSource(ep.ego)
        self.right_src = VideoSource(ep.right_wrist)
        # Per-sample swap is remembered if previously toggled.
        self.swap = bool(self.decisions.get(ep.name, {}).get("swap", False))
        self.frame_pos = 0
        self._load_sim(ep)
        self.episode_badge.setText(f"#{ep.index}")
        self._position_badge()
        self._render()
        self._update_status()

    def _load_sim(self, ep: Episode):
        """Point the MuJoCo replay at the same episode the cameras show."""
        if self.sim is None:
            return
        try:
            self.sim.load_episode(ep.index)
        except Exception as exc:
            # Don't kill the camera viewer if one episode's motion is unreadable.
            self.pane_sim.set_message(f"No sim data for\n{ep.name}:\n{exc}")
            self.sim_loaded = False
        else:
            self.sim_loaded = True

    def _wrist_panes(self):
        """Return (left_source, right_source) honoring the swap toggle."""
        if self.swap:
            return self.right_src, self.left_src
        return self.left_src, self.right_src

    def _render(self):
        left_src, right_src = self._wrist_panes()
        self.pane_left.set_frame(left_src.frame if left_src else None)
        self.pane_ego.set_frame(self.ego_src.frame if self.ego_src else None)
        self.pane_right.set_frame(right_src.frame if right_src else None)
        self.pane_left.set_title("right wrist (swapped)" if self.swap else "left wrist")
        self.pane_right.set_title("left wrist (swapped)" if self.swap else "right wrist")
        self._render_sim()

    def _render_sim(self):
        """Render the MuJoCo G1 at the current frame, kept in sync with video."""
        if self.sim is None or not self.sim_loaded:
            return
        try:
            rgb = self.sim.render(self.frame_pos)
        except Exception as exc:  # pragma: no cover - render-time GL issues
            self.pane_sim.set_message(f"Sim render error:\n{exc}")
            self.sim_loaded = False
            return
        self.pane_sim.set_rgb(rgb)

    def _update_status(self):
        ep = self.episodes[self.idx]
        decision = self.decisions.get(ep.name, {}).get("decision", "—")
        n_good = sum(1 for d in self.decisions.values() if d.get("decision") == "good")
        n_disc = sum(1 for d in self.decisions.values() if d.get("decision") == "discard")
        state = "playing" if self.playing else "paused"
        nframes = self.ego_src.n_frames if self.ego_src else 0
        self.status.setText(
            f"[{self.idx + 1}/{len(self.episodes)}] {ep.name}   "
            f"frame {self.frame_pos}/{max(0, nframes - 1)}   {state}   "
            f"swap={'on' if self.swap else 'off'}   decision={decision}   "
            f"| good={n_good} discard={n_disc}   |   {ep.task}"
        )

    # ---- playback ----------------------------------------------------------
    def _on_tick(self):
        if not self.playing or self.ego_src is None:
            return
        ego_ok = self.ego_src.advance()
        self.left_src.advance()
        self.right_src.advance()
        if not ego_ok:
            # Loop the whole sample back to the start, keeping the feeds aligned.
            for src in (self.left_src, self.ego_src, self.right_src):
                src.restart()
            self.frame_pos = 0
        else:
            self.frame_pos += 1
        self._render()
        self._update_status()

    def _step(self, delta: int):
        if self.ego_src is None:
            return
        self.frame_pos = max(0, self.frame_pos + delta)
        for src in (self.left_src, self.ego_src, self.right_src):
            src.seek(self.frame_pos)
        self._render()
        self._update_status()

    # ---- navigation / decisions -------------------------------------------
    def _goto(self, new_idx: int):
        new_idx = max(0, min(len(self.episodes) - 1, new_idx))
        if new_idx == self.idx and self.left_src is not None:
            return
        self.idx = new_idx
        self._load_current()

    def _record(self, decision: str):
        ep = self.episodes[self.idx]
        entry = self.decisions.get(ep.name, {})
        entry["decision"] = decision
        entry["swap"] = self.swap
        self.decisions[ep.name] = entry
        self._save_decisions()
        if self.idx < len(self.episodes) - 1:
            self._goto(self.idx + 1)
        else:
            self._update_status()  # last sample: stay put, just record

    # ---- keyboard ----------------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self.playing = not self.playing
            self._update_status()
        elif key == QtCore.Qt.Key_Right:
            self.playing = False
            self._step(1)
        elif key == QtCore.Qt.Key_Left:
            self.playing = False
            self._step(-1)
        elif key == QtCore.Qt.Key_S:
            self.swap = not self.swap
            self._render()
            self._update_status()
        elif key == QtCore.Qt.Key_G:
            self._record("good")
        elif key == QtCore.Qt.Key_D:
            self._record("discard")
        elif key in (QtCore.Qt.Key_N, QtCore.Qt.Key_Down):
            self._goto(self.idx + 1)
        elif key in (QtCore.Qt.Key_P, QtCore.Qt.Key_Up):
            self._goto(self.idx - 1)
        elif key == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self._release_sources()
        if self.sim is not None:
            self.sim.close()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tele-op three-camera selector GUI")
    parser.add_argument(
        "--dataset-root",
        default="/home/alain/Documents/red_cube_cardbox_all_cleaned_01",
        help="Path to the LeRobot dataset root.",
    )
    parser.add_argument(
        "--sim-base",
        choices=["upright", "orientation"],
        default="upright",
        help="MuJoCo pane: pin the pelvis upright (default) or apply the "
        "recorded base orientation (relative to frame 0).",
    )
    parser.add_argument(
        "--quat-order",
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="Quaternion order of observation.root_orientation in the dataset.",
    )
    parser.add_argument(
        "--no-sim",
        action="store_true",
        help="Run the camera viewer only, without the MuJoCo G1 replay pane.",
    )
    args = parser.parse_args(argv)

    dataset = Dataset(args.dataset_root)

    app = QtWidgets.QApplication(sys.argv[:1])
    win = SelectorWindow(
        dataset,
        sim_base=args.sim_base,
        quat_order=args.quat_order,
        enable_sim=not args.no_sim,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
