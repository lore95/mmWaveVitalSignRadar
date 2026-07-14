#!/usr/bin/env python3
"""
Recording Plotter — interactive visualization of recorded sessions
──────────────────────────────────────────────────────────────────
Loads a session recorded by record_ui.py and lets you explore:
  • Track metrics over time (BPM, range, angle, validation, SNR)
  • Range-time waterfall from the signals.npz
  • Single-frame ADC trace (if full-mode recording)

Usage:
  python plot_recording.py                                # picks the most recent
  python plot_recording.py --session session_20260706_143022
  python plot_recording.py --dir recordings

The session name is any of the prefixes; the script will find:
  <name>_tracks.csv
  <name>_meta.json
  <name>_signals.npz
"""

import argparse
import json
import os
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from matplotlib.gridspec import GridSpec

# Column display names and units — used in checkbox labels and axis labels
FIELDS = {
    "bpm":              ("BPM",         "breaths/min"),
    "range_m":          ("Range",       "m"),
    "angle_deg":        ("Angle",       "deg"),
    "validated":        ("Validated",   "0/1"),
    "breathing_score":  ("Score",       "0-1"),
    "snr_db":           ("SNR",         "dB"),
}

# Track colors — same palette as the live monitor for consistency
TRACK_COLORS = [
    "#00ff99", "#ff79c6", "#feca57", "#0ff",
    "#ff6361", "#a29bfe", "#fd79a8", "#55efc4",
]


def find_session(sessions_dir, name_hint=None):
    """Locate a session's file trio (tracks.csv, meta.json, signals.npz).

    If name_hint is None, picks the most recent tracks.csv in sessions_dir.
    """
    if name_hint:
        # Strip trailing suffixes if user pasted a full filename
        base = name_hint
        for suffix in ("_tracks.csv", "_meta.json", "_signals.npz", ".csv", ".json", ".npz"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        base = os.path.join(sessions_dir, os.path.basename(base))
    else:
        csv_files = sorted(glob.glob(os.path.join(sessions_dir, "*_tracks.csv")))
        if not csv_files:
            print(f"ERROR: no *_tracks.csv files in {sessions_dir}")
            sys.exit(1)
        base = csv_files[-1][:-len("_tracks.csv")]
        print(f"[LOAD] auto-selected latest session: {os.path.basename(base)}")

    files = {
        "csv":  base + "_tracks.csv",
        "meta": base + "_meta.json",
        "npz":  base + "_signals.npz",
    }
    for kind, path in files.items():
        exists = "✓" if os.path.exists(path) else "✗ missing"
        print(f"        {kind:<5s} {os.path.basename(path)}  {exists}")

    if not os.path.exists(files["csv"]):
        print(f"ERROR: {files['csv']} not found")
        sys.exit(1)

    return files


def load_metadata(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Interactive tracks plotter
# ═══════════════════════════════════════════════════════════════════════

def plot_tracks_interactive(csv_path, meta):
    """Two-panel figure: checkbox column on left, plot on right.

    Each checkbox represents (track_id, field). Toggling redraws the plot
    with only the selected series.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        print("ERROR: no data in CSV")
        return
    track_ids = sorted(df.track_id.unique())
    n_tracks = len(track_ids)
    print(f"[TRACKS] {len(df)} rows, {n_tracks} track(s): {track_ids}")

    # Session duration
    duration_s = float(df.frame_time_s.max())

    # Build a figure with a left panel of checkboxes and right panel of plot
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.canvas.manager.set_window_title(f"Tracks — {os.path.basename(csv_path)}")

    gs = GridSpec(1, 2, width_ratios=[1, 5], wspace=0.15)

    ax_ctrl = fig.add_subplot(gs[0, 0])
    ax_ctrl.set_facecolor("#16213e")
    ax_ctrl.set_xticks([]); ax_ctrl.set_yticks([])
    for sp in ax_ctrl.spines.values():
        sp.set_color("#333")
    ax_ctrl.set_title("plot series", color="white", fontsize=10)

    ax_plot = fig.add_subplot(gs[0, 1])
    ax_plot.set_facecolor("#16213e")
    ax_plot.tick_params(colors="white")
    ax_plot.grid(alpha=0.15, color="white")
    for sp in ax_plot.spines.values():
        sp.set_color("#333")
    ax_plot.set_xlabel("time (s)", color="white")
    ax_plot.set_xlim(0, duration_s + 0.5)

    # Build the checkbox options: one per (track, field) combination.
    # Default: enable BPM for all tracks (the most common thing to look at).
    options = []
    default_active = []
    for tid in track_ids:
        for field, (nice_name, unit) in FIELDS.items():
            label = f"T{tid} · {nice_name}"
            options.append((tid, field, label))
            default_active.append(field == "bpm")

    # Create checkbox widget. Place it inside ax_ctrl.
    ax_ctrl_inner = fig.add_axes([0.02, 0.05, 0.13, 0.90])
    ax_ctrl_inner.set_facecolor("#16213e")
    for sp in ax_ctrl_inner.spines.values():
        sp.set_color("#333")

    labels = [opt[2] for opt in options]
    check = CheckButtons(ax_ctrl_inner, labels, default_active)

    # Style checkbox labels — matplotlib default is black on white
    for lbl, (tid, field, _) in zip(check.labels, options):
        color = TRACK_COLORS[track_ids.index(tid) % len(TRACK_COLORS)]
        lbl.set_color(color)
        lbl.set_fontsize(8)
    # Make the checkbox rectangles visible on dark bg.
    # The attribute name differs across matplotlib versions.
    rects = getattr(check, "rectangles", None) or getattr(check, "_rectangles", None)
    if rects is not None:
        for rect in rects:
            try:
                rect.set_edgecolor("#888")
                rect.set_facecolor("#16213e")
            except Exception:
                pass

    # Metadata / session info text
    notes = meta.get("notes", "")
    duration = meta.get("session", {}).get("duration_s", duration_s)
    mode = meta.get("session", {}).get("mode", "?")
    info_txt = f"session: {duration:.1f}s, mode={mode}"
    if notes:
        info_txt += f", notes: {notes}"
    fig.text(0.20, 0.97, info_txt, color="#aaa", fontsize=9, va="top")

    # Preload per-track data to avoid repeated filtering during redraws
    track_data = {}
    for tid in track_ids:
        d = df[df.track_id == tid].sort_values("frame_time_s")
        track_data[tid] = d

    def redraw():
        ax_plot.clear()
        ax_plot.set_facecolor("#16213e")
        ax_plot.tick_params(colors="white")
        ax_plot.grid(alpha=0.15, color="white")
        for sp in ax_plot.spines.values():
            sp.set_color("#333")
        ax_plot.set_xlabel("time (s)", color="white")

        # Track which axes/units are enabled — for auto-labeling
        active_fields = set()
        n_active = 0

        for i, (tid, field, label) in enumerate(options):
            if not check.get_status()[i]:
                continue
            d = track_data[tid]
            if d.empty:
                continue
            color = TRACK_COLORS[track_ids.index(tid) % len(TRACK_COLORS)]
            # Different line styles for different fields
            linestyle = {
                "bpm":             "-",
                "range_m":         "--",
                "angle_deg":       ":",
                "validated":       "-",
                "breathing_score": "-.",
                "snr_db":          (0, (3, 1, 1, 1)),  # dash-dot-dot
            }.get(field, "-")

            y = d[field].values
            ax_plot.plot(
                d.frame_time_s.values, y,
                color=color, linewidth=1.4, alpha=0.9,
                linestyle=linestyle,
                label=label,
            )
            active_fields.add(field)
            n_active += 1

        # Y-axis label — if only one field type is plotted, use its unit
        if len(active_fields) == 1:
            f = next(iter(active_fields))
            nice_name, unit = FIELDS[f]
            ax_plot.set_ylabel(f"{nice_name} ({unit})", color="white")
        elif len(active_fields) > 1:
            ax_plot.set_ylabel("value (mixed units)", color="white")
        else:
            ax_plot.set_ylabel("(nothing selected)", color="#666")

        # Legend
        if n_active > 0:
            leg = ax_plot.legend(
                loc="upper right", fontsize=8,
                facecolor="#16213e", edgecolor="#333",
                labelcolor="white", ncol=max(1, n_active // 6))

        ax_plot.set_xlim(0, duration_s + 0.5)
        fig.canvas.draw_idle()

    check.on_clicked(lambda _label: redraw())
    redraw()

    # Buttons: select-all, clear-all, presets
    ax_btn_all = fig.add_axes([0.02, 0.005, 0.06, 0.03])
    btn_all = Button(ax_btn_all, "all", color="#333", hovercolor="#555")
    btn_all.label.set_color("white"); btn_all.label.set_fontsize(8)

    ax_btn_clear = fig.add_axes([0.09, 0.005, 0.06, 0.03])
    btn_clear = Button(ax_btn_clear, "clear", color="#333", hovercolor="#555")
    btn_clear.label.set_color("white"); btn_clear.label.set_fontsize(8)

    def set_all(state):
        # CheckButtons has no direct set-state API — flip whichever mismatch
        current = check.get_status()
        for i, cur in enumerate(current):
            if cur != state:
                check.set_active(i)  # toggles

    btn_all.on_clicked(lambda _: (set_all(True), redraw()))
    btn_clear.on_clicked(lambda _: (set_all(False), redraw()))


# ═══════════════════════════════════════════════════════════════════════
# Range-time waterfall from signals.npz
# ═══════════════════════════════════════════════════════════════════════

def plot_waterfall(npz_path, meta):
    """Range-time heatmap (waterfall) of the recorded range profiles.

    Each column of the image is one frame's range profile in dB. Reading
    it top-to-bottom shows what was at each range bin at that moment.
    Time flows left→right.
    """
    if not os.path.exists(npz_path):
        print(f"[SKIP] no signals file at {npz_path}")
        return

    data = np.load(npz_path)
    if "range_profiles" not in data.files:
        print(f"[SKIP] no range_profiles in {npz_path}")
        return

    rp = data["range_profiles"]      # (n_frames, n_bins), float32 magnitude
    t = data["frame_times"]           # (n_frames,)
    n_frames, n_bins = rp.shape
    print(f"[WATERFALL] {n_frames} frames × {n_bins} range bins")

    # Compute the range axis assuming defaults; also try to override from meta
    slope = 78.020e12
    fs = 10e6
    if "radar" in meta:
        slope = float(meta["radar"].get("slope_hz_per_s", slope))
        fs = float(meta["radar"].get("sample_rate_hz", fs))
    C = 2.998e8
    n_adc = n_bins * 2  # range profile has N/2 bins
    rng = np.arange(n_bins) * (fs / n_adc) * C / (2 * slope)

    # Convert to dB, clip to reasonable range
    rp_db = 20 * np.log10(np.maximum(np.abs(rp), 1e-6))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.canvas.manager.set_window_title("Range-time waterfall")
    ax.set_facecolor("#0a0a1a")

    # Show as image: rows = range (increasing downward), columns = time
    # extent: (x_min, x_max, y_min, y_max), origin='lower' for range up
    im = ax.imshow(
        rp_db.T,                       # transpose so range is vertical
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=(t.min(), t.max(), rng.min(), rng.max()),
        vmin=np.percentile(rp_db, 5),
        vmax=np.percentile(rp_db, 99),
    )
    ax.set_xlabel("time (s)", color="white")
    ax.set_ylabel("range (m)", color="white")
    ax.set_title("range-time waterfall (background-subtracted, dB)",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#333")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("dB", color="white")
    cbar.ax.tick_params(colors="white")


# ═══════════════════════════════════════════════════════════════════════
# Raw ADC single-frame viewer (full mode only)
# ═══════════════════════════════════════════════════════════════════════

def plot_raw_adc(npz_path):
    """Slider-driven viewer for a single frame's raw ADC time series.

    Only available when the recording was made in 'full' mode. Lets you
    pick a frame index, chirp index, and RX channel, and shows the I/Q
    time series for that chirp. Rarely useful except for debugging bad
    frames or verifying the parser is working.
    """
    if not os.path.exists(npz_path):
        print(f"[SKIP] no signals file at {npz_path}")
        return
    data = np.load(npz_path)
    if "raw_adc" not in data.files:
        print("[SKIP] raw ADC not in signals (recording was summary mode)")
        return

    raw = data["raw_adc"]
    layout = data["frame_layout"]      # [TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES]
    total_chirps, num_rx, num_samples = int(layout[0]), int(layout[1]), int(layout[2])
    bytes_per_frame_int16 = total_chirps * num_rx * num_samples * 2
    n_frames = len(raw) // bytes_per_frame_int16
    print(f"[RAW] {n_frames} frames of raw ADC, "
          f"({total_chirps}×{num_rx}×{num_samples})")

    def parse_frame(frame_idx):
        """Extract one frame from the concatenated raw array (LVDS-interleaved)."""
        chunk = raw[frame_idx * bytes_per_frame_int16:
                    (frame_idx + 1) * bytes_per_frame_int16]
        ret = np.zeros(len(chunk) // 2, dtype=np.complex64)
        ret[0::2] = chunk[0::4].astype(np.float32) + 1j * chunk[2::4].astype(np.float32)
        ret[1::2] = chunk[1::4].astype(np.float32) + 1j * chunk[3::4].astype(np.float32)
        return ret.reshape(total_chirps, num_rx, num_samples)

    # Fig with sliders
    from matplotlib.widgets import Slider
    fig, (ax_iq, ax_fft) = plt.subplots(2, 1, figsize=(11, 7))
    fig.patch.set_facecolor("#1a1a2e")
    fig.canvas.manager.set_window_title("Raw ADC viewer")
    plt.subplots_adjust(left=0.10, right=0.95, top=0.93, bottom=0.20, hspace=0.35)

    for ax in (ax_iq, ax_fft):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.15, color="white")
        for sp in ax.spines.values():
            sp.set_color("#333")

    ax_iq.set_title("raw ADC I/Q trace", color="white")
    ax_iq.set_xlabel("sample", color="white")
    ax_iq.set_ylabel("amplitude", color="white")
    ax_fft.set_title("range FFT (magnitude, dB)", color="white")
    ax_fft.set_xlabel("range bin", color="white")
    ax_fft.set_ylabel("dB", color="white")

    line_i, = ax_iq.plot([], [], color="#00ff99", lw=1, label="I")
    line_q, = ax_iq.plot([], [], color="#ff79c6", lw=1, label="Q")
    line_fft, = ax_fft.plot([], [], color="#feca57", lw=1)
    ax_iq.legend(loc="upper right", facecolor="#16213e",
                 edgecolor="#333", labelcolor="white")

    # Sliders
    ax_frame = plt.axes([0.10, 0.10, 0.75, 0.02], facecolor="#16213e")
    ax_chirp = plt.axes([0.10, 0.06, 0.75, 0.02], facecolor="#16213e")
    ax_rx    = plt.axes([0.10, 0.02, 0.75, 0.02], facecolor="#16213e")

    s_frame = Slider(ax_frame, "frame", 0, n_frames - 1, valinit=0, valstep=1,
                     color="#00ff99")
    s_chirp = Slider(ax_chirp, "chirp", 0, total_chirps - 1, valinit=0, valstep=1,
                     color="#ff79c6")
    s_rx    = Slider(ax_rx,    "rx",    0, num_rx - 1,        valinit=0, valstep=1,
                     color="#feca57")

    for s in (s_frame, s_chirp, s_rx):
        s.label.set_color("white")
        s.valtext.set_color("white")

    def update(_):
        fi = int(s_frame.val)
        ci = int(s_chirp.val)
        ri = int(s_rx.val)
        frame = parse_frame(fi)
        trace = frame[ci, ri, :]
        x = np.arange(num_samples)
        line_i.set_data(x, trace.real)
        line_q.set_data(x, trace.imag)
        ax_iq.relim(); ax_iq.autoscale_view()

        # Range FFT
        w = np.hanning(num_samples)
        dc = trace.mean()
        rfft = np.fft.fft((trace - dc) * w)[:num_samples // 2]
        mag_db = 20 * np.log10(np.abs(rfft) + 1e-6)
        line_fft.set_data(np.arange(len(mag_db)), mag_db)
        ax_fft.relim(); ax_fft.autoscale_view()

        fig.canvas.draw_idle()

    s_frame.on_changed(update)
    s_chirp.on_changed(update)
    s_rx.on_changed(update)
    update(None)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Plot a recorded session")
    ap.add_argument("--dir", default="recordings", help="recordings directory")
    ap.add_argument("--session", default=None,
                    help="session name prefix (e.g. session_20260706_143022). "
                         "If omitted, the most recent session is used.")
    ap.add_argument("--no-tracks", action="store_true")
    ap.add_argument("--no-waterfall", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="also open raw ADC viewer (full-mode recordings only)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: {args.dir} does not exist")
        sys.exit(1)

    files = find_session(args.dir, args.session)
    meta = load_metadata(files["meta"])

    if not args.no_tracks:
        plot_tracks_interactive(files["csv"], meta)

    if not args.no_waterfall:
        plot_waterfall(files["npz"], meta)

    if args.raw:
        plot_raw_adc(files["npz"])

    plt.show()


if __name__ == "__main__":
    main()