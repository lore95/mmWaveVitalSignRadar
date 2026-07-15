#!/usr/bin/env python3
"""
Offline Vital Sign Detection — replay a recorded session
─────────────────────────────────────────────────────────
Reads a session's signals.npz file and replays the recorded complex range
FFTs through the same processing pipeline as the live monitor:
   background subtraction → CFAR peak detection → Kalman tracking →
   phase extraction → BPM estimation → sinusoidal validation → AoA

Because the range FFT is already computed and saved, this replay is fast:
you can iterate on tracker parameters, validation thresholds, and AoA
algorithms without needing the radar hardware or the raw ADC data.

Usage:
  python offline_vital_sign_detection.py                        # latest session
  python offline_vital_sign_detection.py --session NAME         # named session
  python offline_vital_sign_detection.py --dir recordings/foo   # explicit path
  python offline_vital_sign_detection.py --speed 4              # 4x real-time
  python offline_vital_sign_detection.py --dump-csv out.csv     # non-interactive

What you can vary offline (in this script's constants section):
  • CFAR_K, SNR_THRESHOLD_DB (detection thresholds)
  • VALIDATION_ENERGY, VALIDATION_CONCENT, VALIDATION_SNR_DB (breathing filter)
  • Kalman noise & gate parameters
  • Adaptive background settings

What you canNOT vary offline (baked into the recording):
  • Range FFT window / DC removal
  • Chirp-level coherent averaging (64 chirps averaged into each frame)
  • Sample rate, chirp slope, ADC samples (radar-level RF settings)
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge

from multi_track_manager import MultiTrackManager, TRACK_COLORS


# ═══════════════════════════════════════════════════════════════════════
# Constants — mirror the live monitor. Tweak these for offline experiments.
# ═══════════════════════════════════════════════════════════════════════
# These MUST match the config used during recording (they describe the data
# that's already in the NPZ, not our processing choices).
NUM_ADC_SAMPLES   = 256
FRAME_PERIOD_S    = 20e-3
FPS_HZ            = 1.0 / FRAME_PERIOD_S

C                 = 2.998e8
SLOPE_HZ_PER_S    = 78.020e12
SAMPLE_RATE_HZ    = 10e6
F0_HZ             = 77e9
LAMBDA_M          = C / F0_HZ
AOA_FFT_SIZE      = 64

# Number of virtual elements per frame (3 TX × 4 RX)
NUM_VIRTUAL       = 12

# These are the SOFTWARE parameters — the point of the replay is that you
# can tweak them and re-run to see how the outputs change.
BG_FRAMES         = 500
BP_LO, BP_HI      = 0.04, 0.6
BPM_WINDOW_S      = 40
BPM_REFRESH_S     = 5.0

MAX_TRACKS        = 4
SEARCH_MIN_M      = 0.7
SEARCH_MAX_M      = 4.0
CONFIRM_FRAMES    = 3
DELETE_FRAMES     = 150
MIN_PEAK_SEP_BINS = 6
MIN_TRACK_SEP_BINS = 4
SNR_THRESHOLD_DB  = 15.0
CFAR_K            = 7.0
KALMAN_Q_POS      = 0.1
KALMAN_Q_VEL      = 0.01
KALMAN_R_MEAS     = 1.0
KALMAN_GATE_SIGMA = 5.0

XMAX_M            = 4.0
ANGLE_MAX_DEG     = 60

# Where to look
DEFAULT_RECORDINGS_DIR = "recordings"


# ═══════════════════════════════════════════════════════════════════════
# Helpers (mirror live monitor)
# ═══════════════════════════════════════════════════════════════════════
def range_axis(n):
    return np.arange(n // 2) * (SAMPLE_RATE_HZ / n) * C / (2 * SLOPE_HZ_PER_S)


def angle_axis(n_fft=AOA_FFT_SIZE):
    u = np.linspace(-1, 1, n_fft, endpoint=False)
    return np.degrees(np.arcsin(np.clip(u, -1, 1)))


def compute_range_angle_map(virtual, n_fft=AOA_FFT_SIZE):
    """Angle FFT across the virtual dimension → (n_fft, n_bins) magnitude dB.

    The caller decides which subset of virtual elements is azimuth.
    """
    angle_fft = np.fft.fftshift(np.fft.fft(virtual, n=n_fft, axis=0), axes=0)
    mag = np.abs(angle_fft)
    return 20 * np.log10(mag + 1e-6)


def estimate_aoa(rfft_tx0, rfft_tx2, bin_idx, n_fft=AOA_FFT_SIZE):
    idx = min(bin_idx, rfft_tx0.shape[1] - 1)
    virtual = np.zeros(8, dtype=np.complex64)
    virtual[0:4] = rfft_tx0[:, idx]
    virtual[4:8] = rfft_tx2[:, idx]
    spectrum = np.fft.fftshift(np.fft.fft(virtual, n=n_fft))
    angles = angle_axis(n_fft)
    return float(angles[int(np.argmax(np.abs(spectrum)))])


# ═══════════════════════════════════════════════════════════════════════
# Session loading
# ═══════════════════════════════════════════════════════════════════════
def find_session(sessions_dir, name_hint=None):
    """Find the trio of files for a session.

    Handles both flat layout (recordings/session_X_tracks.csv, etc.) and
    nested layout (recordings/session_X/session_X_tracks.csv, etc.).
    """
    if name_hint:
        # Strip any trailing suffix the user might have typed
        base = name_hint
        for sfx in ("_tracks.csv", "_meta.json", "_signals.npz",
                    ".csv", ".json", ".npz"):
            if base.endswith(sfx):
                base = base[:-len(sfx)]
        # Try nested layout first: recordings_dir/base/base_signals.npz
        nested = os.path.join(sessions_dir, os.path.basename(base),
                              os.path.basename(base))
        flat = os.path.join(sessions_dir, os.path.basename(base))
        if os.path.exists(nested + "_signals.npz"):
            return nested
        if os.path.exists(flat + "_signals.npz"):
            return flat
        print(f"ERROR: could not find _signals.npz for '{name_hint}'")
        sys.exit(1)

    # Auto-pick most recent — search both flat and nested layouts
    candidates = (
        sorted(glob.glob(os.path.join(sessions_dir, "**", "*_signals.npz"),
                         recursive=True))
    )
    if not candidates:
        print(f"ERROR: no *_signals.npz found in {sessions_dir}")
        sys.exit(1)
    latest = candidates[-1]
    base = latest[:-len("_signals.npz")]
    print(f"[LOAD] auto-selected latest session: {os.path.basename(base)}")
    return base


def load_session(base_path):
    files = {
        "npz":  base_path + "_signals.npz",
        "meta": base_path + "_meta.json",
        "csv":  base_path + "_tracks.csv",
    }
    for kind, path in files.items():
        print(f"        {kind:<5s} {os.path.basename(path)}  "
              f"{'✓' if os.path.exists(path) else '✗ missing'}")

    if not os.path.exists(files["npz"]):
        print(f"ERROR: signals file missing")
        sys.exit(1)

    npz = np.load(files["npz"])
    if "range_complex" not in npz.files:
        print(f"ERROR: this recording predates offline replay support")
        print(f"       (no 'range_complex' array in the NPZ file).")
        print(f"       Record a new session with the updated record_ui.py.")
        sys.exit(1)

    meta = {}
    if os.path.exists(files["meta"]):
        with open(files["meta"]) as f:
            meta = json.load(f)

    range_complex = npz["range_complex"]     # (n_frames, 12, n_bins) complex64
    frame_times = npz["frame_times"]          # (n_frames,) float64
    print(f"[LOAD] {range_complex.shape[0]} frames, "
          f"virtual shape {range_complex.shape[1:]}")
    print(f"[LOAD] duration: {frame_times[-1]:.1f} s")

    return {
        "range_complex": range_complex,
        "frame_times": frame_times,
        "meta": meta,
        "base": base_path,
    }


# ═══════════════════════════════════════════════════════════════════════
# The replay engine
# ═══════════════════════════════════════════════════════════════════════
class ReplayEngine:
    """Feeds recorded frames through the live-monitor pipeline.

    Owns:
      - the recorded range_complex array
      - background stats computed from the first BG_FRAMES frames
      - a MultiTrackManager configured with current-file parameters
      - a per-frame history of track states (built as replay progresses)
    """

    def __init__(self, session, playback_speed=1.0):
        self.session = session
        self.range_complex = session["range_complex"]
        self.frame_times = session["frame_times"]
        self.n_frames = len(self.frame_times)
        self.n_bins = self.range_complex.shape[2]
        self.speed = playback_speed

        # Set up range/angle axes
        self.rng = range_axis(NUM_ADC_SAMPLES)
        self.search_min_bin = max(1, int(np.searchsorted(self.rng, SEARCH_MIN_M)))
        self.search_max_bin = min(self.n_bins - 1,
                                  int(np.searchsorted(self.rng, SEARCH_MAX_M)))
        self.angles = angle_axis(AOA_FFT_SIZE)
        self.keep_mask = self.rng <= XMAX_M
        self.angle_mask = (np.abs(self.angles) <= ANGLE_MAX_DEG)

        # Background stats (computed from the first BG_FRAMES frames of
        # the recording — this mirrors what the live monitor does).
        bg_end = min(BG_FRAMES, self.n_frames)
        bg_data = self.range_complex[:bg_end]          # (bg_end, 12, n_bins)
        self.bg_virtual = bg_data.mean(axis=0)           # (12, n_bins) complex

        # Per-bin noise stats from the (already averaged over virtual) profile
        # Same as live: median magnitude across virtual dim, std across frames
        avg_over_virtual = bg_data.mean(axis=1)          # (bg_end, n_bins)
        self.bg_noise_std = np.std(np.abs(avg_over_virtual), axis=0)

        print(f"[REPLAY] background computed from first {bg_end} frames")
        print(f"[REPLAY] noise floor σ  median={np.median(self.bg_noise_std):.1f}")

        # Track manager
        self.track_mgr = MultiTrackManager(
            range_axis=self.rng, lambda_m=LAMBDA_M, fps=FPS_HZ,
            min_bin=self.search_min_bin, max_bin=self.search_max_bin,
            max_tracks=MAX_TRACKS,
            confirm_frames=CONFIRM_FRAMES,
            delete_frames=DELETE_FRAMES,
            min_peak_separation_bins=MIN_PEAK_SEP_BINS,
            min_track_separation_bins=MIN_TRACK_SEP_BINS,
            snr_threshold_db=SNR_THRESHOLD_DB,
            bpm_window_s=BPM_WINDOW_S,
            bpm_refresh_s=BPM_REFRESH_S,
            q_pos=KALMAN_Q_POS, q_vel=KALMAN_Q_VEL,
            r_meas=KALMAN_R_MEAS, gate_sigma=KALMAN_GATE_SIGMA,
        )
        self.track_mgr.set_background_stats(self.bg_noise_std, cfar_k=CFAR_K)

        # State
        self.cursor = bg_end   # start replaying AFTER the calibration window
        self.track_angles = {}
        self.last_ra_map_db = None
        self.last_diff_avg = None

    def rewind(self):
        """Reset to the start of the post-calibration window and rebuild manager."""
        self.cursor = min(BG_FRAMES, self.n_frames)
        self.track_angles = {}
        self.track_mgr = MultiTrackManager(
            range_axis=self.rng, lambda_m=LAMBDA_M, fps=FPS_HZ,
            min_bin=self.search_min_bin, max_bin=self.search_max_bin,
            max_tracks=MAX_TRACKS,
            confirm_frames=CONFIRM_FRAMES,
            delete_frames=DELETE_FRAMES,
            min_peak_separation_bins=MIN_PEAK_SEP_BINS,
            min_track_separation_bins=MIN_TRACK_SEP_BINS,
            snr_threshold_db=SNR_THRESHOLD_DB,
            bpm_window_s=BPM_WINDOW_S,
            bpm_refresh_s=BPM_REFRESH_S,
            q_pos=KALMAN_Q_POS, q_vel=KALMAN_Q_VEL,
            r_meas=KALMAN_R_MEAS, gate_sigma=KALMAN_GATE_SIGMA,
        )
        self.track_mgr.set_background_stats(self.bg_noise_std, cfar_k=CFAR_K)

    def step(self):
        """Advance one frame. Returns False when the session ends."""
        if self.cursor >= self.n_frames:
            return False

        i = self.cursor
        self.cursor += 1

        virtual = self.range_complex[i]         # (12, n_bins) complex
        t_rel = float(self.frame_times[i])

        # Background subtraction — for detection path (magnitude only)
        diff_virtual = virtual - self.bg_virtual
        diff_avg = diff_virtual.mean(axis=0)     # (n_bins,) complex

        # Detection uses magnitude of bg-subtracted signal
        mag_det = np.abs(diff_avg)

        # Phase extraction uses RAW complex data at the 8 azimuth elements
        # (TX0 = elements 0-3, TX2 = elements 8-11 in our layout)
        rfft_tx0 = virtual[0:4]
        rfft_tx2 = virtual[8:12]
        azimuth_raw = np.concatenate([rfft_tx0, rfft_tx2], axis=0)  # (8, n_bins)
        rfft_avg_raw = azimuth_raw.mean(axis=0)   # complex (n_bins,)

        # Feed the tracker
        self.track_mgr.step(mag_det, rfft_avg_raw, t_rel)

        # AoA per track — using bg-subtracted azimuth for cleaner angle spectrum
        azimuth_diff = np.concatenate([diff_virtual[0:4], diff_virtual[8:12]], axis=0)
        for trk in self.track_mgr.confirmed_tracks:
            bi = int(round(trk.bin_position))
            self.track_angles[trk.track_id] = estimate_aoa(
                azimuth_diff[0:4], azimuth_diff[4:8], bi)

        # Store range-angle map for display
        self.last_ra_map_db = compute_range_angle_map(azimuth_diff)
        self.last_diff_avg = mag_det

        return True

    def run_to_end(self):
        """Process everything without any UI (fast, for batch analysis)."""
        while self.step():
            pass


# ═══════════════════════════════════════════════════════════════════════
# CSV export (non-interactive mode)
# ═══════════════════════════════════════════════════════════════════════
def dump_tracks_to_csv(engine, out_path):
    """Run the engine to completion, capturing per-frame track state,
    and write a tracks CSV in the same format as record_ui.py.

    Since this replays a completed recording, we can't stream — we run to
    the end and buffer track states.
    """
    import csv

    rows = []
    while True:
        i_before = engine.cursor
        if not engine.step():
            break
        i = i_before   # frame index we just processed

        for trk in engine.track_mgr.confirmed_tracks:
            rows.append([
                i,
                float(engine.frame_times[i]),
                int(trk.track_id),
                float(trk.range_m),
                float(engine.track_angles.get(trk.track_id, 0.0)),
                float(trk.bpm),
                1 if trk.breathing_validated else 0,
                float(getattr(trk, "breathing_score", 0.0)),
                float(getattr(trk, "peak_snr_db", 0.0)),
            ])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "frame_time_s", "track_id", "range_m",
                    "angle_deg", "bpm", "validated",
                    "breathing_score", "snr_db"])
        for row in rows:
            w.writerow([row[0], f"{row[1]:.4f}", row[2],
                        f"{row[3]:.3f}", f"{row[4]:.2f}", f"{row[5]:.2f}",
                        row[6], f"{row[7]:.4f}", f"{row[8]:.2f}"])

    print(f"[CSV] wrote {len(rows)} track observations to {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Interactive playback UI
# ═══════════════════════════════════════════════════════════════════════
def run_interactive(engine, session):
    """Live-monitor-style UI, but time is driven by playback of the recording.

    Includes:
      - the fan (range-angle heatmap) with track wedges
      - BPM readouts per track
      - a time slider you can drag to any point
      - play / pause / rewind buttons
      - speed multiplier
    """
    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.canvas.manager.set_window_title(
        f"Offline replay — {os.path.basename(session['base'])}")

    gs = GridSpec(4, 3, figure=fig,
                  height_ratios=[8, 0.5, 0.5, 0.5],
                  width_ratios=[3, 3, 1],
                  hspace=0.5, wspace=0.25,
                  left=0.06, right=0.97, top=0.94, bottom=0.06)

    # Fan display
    ax_ra = fig.add_subplot(gs[0, 0:2])
    ax_ra.set_facecolor("#0a0a1a")
    ax_ra.set_xlabel("lateral (m)", color="white")
    ax_ra.set_ylabel("range (m)", color="white")
    ax_ra.set_title(
        f"offline replay ({os.path.basename(session['base'])})",
        color="white", fontsize=11)
    ax_ra.tick_params(colors="white")
    ax_ra.set_xlim(-XMAX_M * 0.87, XMAX_M * 0.87)
    ax_ra.set_ylim(0, XMAX_M)
    ax_ra.set_aspect("equal")

    # Pre-compute the mesh for the fan (cartesian coords for each range-angle cell)
    rng_keep = engine.rng[engine.keep_mask]
    ang_keep = engine.angles[engine.angle_mask]
    R, A = np.meshgrid(rng_keep, np.radians(ang_keep))
    X = R * np.sin(A)
    Y = R * np.cos(A)
    # Initial empty heatmap
    dummy = np.full((len(ang_keep), len(rng_keep)), 40.0)
    pcm = ax_ra.pcolormesh(X, Y, dummy, cmap="turbo",
                           shading="nearest", vmin=40, vmax=90, zorder=1)
    cbar = fig.colorbar(pcm, ax=ax_ra, pad=0.02, fraction=0.04)
    cbar.set_label("dB", color="white")
    cbar.ax.tick_params(colors="white")

    # Track wedges (kept as a list so we can clear/redraw)
    track_patches = []

    # BPM sidebar
    ax_side = fig.add_subplot(gs[0, 2])
    ax_side.set_facecolor("#16213e")
    ax_side.set_xticks([]); ax_side.set_yticks([])
    for sp in ax_side.spines.values():
        sp.set_color("#333")
    ax_side.set_title("BPM", color="white", fontsize=10)

    bpm_texts = []
    for i in range(MAX_TRACKS):
        y = 0.85 - i * 0.22
        color = TRACK_COLORS[i]
        ax_side.text(0.5, y + 0.08, f"track {i+1}",
                     transform=ax_side.transAxes, ha="center",
                     color=color, fontsize=9)
        bt = ax_side.text(0.5, y, "--",
                          transform=ax_side.transAxes, ha="center",
                          color="#333", fontsize=22,
                          family="monospace", fontweight="bold")
        bpm_texts.append(bt)

    # Status line above fan
    txt_status = ax_ra.text(
        0.98, 0.95, "", transform=ax_ra.transAxes, ha="right", va="top",
        color="#ff9f43", fontsize=10)

    # Playback controls
    total_frames = engine.n_frames - BG_FRAMES
    ax_slider = fig.add_subplot(gs[1, 0:2])
    ax_slider.set_facecolor("#16213e")
    slider = Slider(ax_slider, "time",
                    valmin=0.0, valmax=float(engine.frame_times[-1]),
                    valinit=float(engine.frame_times[BG_FRAMES]),
                    color="#00ff99")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    ax_play = fig.add_subplot(gs[2, 0])
    ax_pause = fig.add_subplot(gs[2, 1])
    ax_rewind = fig.add_subplot(gs[3, 0])
    ax_speed = fig.add_subplot(gs[3, 1])

    btn_play = Button(ax_play, "▶ play", color="#00b050", hovercolor="#00d060")
    btn_pause = Button(ax_pause, "⏸ pause", color="#333", hovercolor="#555")
    btn_rewind = Button(ax_rewind, "⟲ rewind", color="#333", hovercolor="#555")
    for b in (btn_play, btn_pause, btn_rewind):
        b.label.set_color("white")

    # Speed slider
    ax_speed.set_facecolor("#16213e")
    speed_slider = Slider(ax_speed, "speed", 0.25, 8.0,
                          valinit=engine.speed, valstep=0.25,
                          color="#ff79c6")
    speed_slider.label.set_color("white")
    speed_slider.valtext.set_color("white")

    # State
    state = {"playing": True, "last_wall": time.monotonic()}

    def on_play(_):
        state["playing"] = True
        state["last_wall"] = time.monotonic()

    def on_pause(_):
        state["playing"] = False

    def on_rewind(_):
        state["playing"] = False
        engine.rewind()
        slider.set_val(float(engine.frame_times[BG_FRAMES]))

    def on_speed(v):
        engine.speed = float(v)

    def on_slider(v):
        # User dragged the slider — seek there and pause
        state["playing"] = False
        t_target = float(v)
        # Find nearest frame index and reset tracker up to that point
        target_idx = int(np.searchsorted(engine.frame_times, t_target))
        target_idx = max(BG_FRAMES, min(engine.n_frames - 1, target_idx))
        # Simplest correct behavior: rewind and step forward to target
        engine.rewind()
        while engine.cursor < target_idx:
            engine.step()

    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    btn_rewind.on_clicked(on_rewind)
    speed_slider.on_changed(on_speed)
    # Only react to programmatic vs user-dragged slider changes by
    # detecting whether the caller was us — matplotlib doesn't distinguish.
    # We accept that when we update the slider ourselves during playback,
    # we don't re-run the tracker. Attach the callback ONLY for user drags:
    #    matplotlib doesn't expose this, so we suppress redundant work by
    #    checking if the target matches the current cursor.
    def on_slider_maybe(v):
        target_idx = int(np.searchsorted(engine.frame_times, float(v)))
        target_idx = max(BG_FRAMES, min(engine.n_frames - 1, target_idx))
        if abs(target_idx - engine.cursor) > 2:
            on_slider(v)

    slider.on_changed(on_slider_maybe)

    def render():
        """Update all visual elements from the engine's current state."""
        # Clear old track wedges
        for p in track_patches:
            p.remove()
        track_patches.clear()

        # Update fan
        if engine.last_ra_map_db is not None:
            ra_show = engine.last_ra_map_db[engine.angle_mask][:, engine.keep_mask]
            pcm.set_array(ra_show.ravel())

        # Update tracks
        confirmed = engine.track_mgr.confirmed_tracks
        n_conf = len(confirmed)
        n_val = sum(1 for t in confirmed if t.breathing_validated)

        for t in confirmed:
            a = engine.track_angles.get(t.track_id, 0)
            r = t.range_m
            if t.breathing_validated:
                beta = 90.0 - a
                wedge = Wedge(
                    center=(0, 0), r=r + 0.15,
                    theta1=beta - 6, theta2=beta + 6,
                    width=0.30, facecolor=t.color, edgecolor="white",
                    linewidth=1.5, alpha=0.85, zorder=5)
                ax_ra.add_patch(wedge)
                track_patches.append(wedge)

        # Status text
        parts = []
        for t in confirmed:
            a = engine.track_angles.get(t.track_id, 0)
            state_char = "✓" if t.breathing_validated else "…"
            parts.append(f"{state_char}{t.range_m:.1f}m/{a:+.0f}°")
        if parts:
            txt_status.set_text(f"{n_val} breathing, {n_conf-n_val} pending: {', '.join(parts)}")
        else:
            txt_status.set_text("no targets")

        # BPM readouts
        for i, bt in enumerate(bpm_texts):
            if i < n_conf:
                t = confirmed[i]
                if t.breathing_validated:
                    bt.set_color(t.color)
                    bt.set_text(f"{t.bpm:.0f}" if t.bpm > 0 else "--")
                else:
                    bt.set_color("#555")
                    bt.set_text("…")
            else:
                bt.set_text("--")
                bt.set_color("#333")

        # Slider position — programmatic update
        if engine.cursor - 1 < engine.n_frames:
            t_cur = float(engine.frame_times[engine.cursor - 1])
            # Update without triggering our on-drag handler
            slider.eventson = False
            slider.set_val(t_cur)
            slider.eventson = True

    def update(_):
        if state["playing"]:
            now = time.monotonic()
            dt_wall = now - state["last_wall"]
            state["last_wall"] = now
            # How many frames to advance? engine.speed × wall_dt / frame_period
            n_advance = max(1, int(dt_wall * engine.speed / FRAME_PERIOD_S))
            for _ in range(n_advance):
                if not engine.step():
                    state["playing"] = False
                    break
        render()

    # Advance one frame immediately so first render has data
    engine.step()

    ani = FuncAnimation(fig, update, interval=50, blit=False,
                        cache_frame_data=False)

    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Offline vital-sign replay from a recorded session")
    ap.add_argument("--dir", default=DEFAULT_RECORDINGS_DIR,
                    help="recordings directory")
    ap.add_argument("--session", default=None,
                    help="session name prefix (default: latest)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="initial playback speed multiplier (1.0 = real-time)")
    ap.add_argument("--dump-csv", default=None,
                    help="write replayed tracks to this CSV file and exit "
                         "(no UI)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: {args.dir} does not exist")
        sys.exit(1)

    base = find_session(args.dir, args.session)
    session = load_session(base)
    session["base"] = base

    engine = ReplayEngine(session, playback_speed=args.speed)

    if args.dump_csv:
        print(f"[BATCH] running replay to end, dumping to {args.dump_csv}")
        dump_tracks_to_csv(engine, args.dump_csv)
        return

    run_interactive(engine, session)


if __name__ == "__main__":
    main()
