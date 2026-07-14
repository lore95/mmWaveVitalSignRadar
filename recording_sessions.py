#!/usr/bin/env python3
"""
Recording UI — minimal breathing monitor + data logger
──────────────────────────────────────────────────────
A stripped-down version of the main monitor with:
  • Small UI showing number of tracked people and per-track BPM
  • Two buttons: Start / Stop recording
  • Two recording modes:
      Summary:  ~1 MB / minute — track states + range profiles + metadata
      Full:     ~2.3 GB / minute — raw ADC frames + everything above
  • Recording goes into ./recordings/ named with timestamp

Recorded files are .npz archives that can be loaded with numpy.

Usage:
  python record_ui.py
  python record_ui.py --no-radar --belt-only  # for testing UI without radar
"""

import argparse
import threading
import socket
import time
import os
import sys
import collections
import json
import csv
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons

from kalman_tracker import KalmanPeakTracker
from multi_track_manager import MultiTrackManager, TRACK_COLORS

# ═══════════════════════════════════════════════════════════════════════
# Radar constants (match Lua config)
# ═══════════════════════════════════════════════════════════════════════
NUM_RX            = 4
NUM_TX            = 3
NUM_CHIRPS        = 64
NUM_ADC_SAMPLES   = 256
FRAME_PERIOD_S    = 20e-3
FPS_HZ            = 1.0 / FRAME_PERIOD_S

C                 = 2.998e8
SLOPE_HZ_PER_S    = 78.020e12
SAMPLE_RATE_HZ    = 10e6
F0_HZ             = 77e9
LAMBDA_M          = C / F0_HZ

TOTAL_CHIRPS      = NUM_CHIRPS * NUM_TX
COMPLEX_PER_FRAME = TOTAL_CHIRPS * NUM_RX * NUM_ADC_SAMPLES
RAW_BYTES_PER_FRAME = COMPLEX_PER_FRAME * 4
AOA_FFT_SIZE      = 64

BG_FRAMES         = 500
BP_LO, BP_HI      = 0.04, 0.6
BPM_WINDOW_S      = 40
BPM_REFRESH_S     = 5.0

HOST_IP           = "192.168.33.30"
DATA_PORT         = 4098
CALIBRATION_FILE  = "radar_phase_calibration.npz"
RECORDING_DIR     = "recordings"

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


# ═══════════════════════════════════════════════════════════════════════
# Radar helpers (mirror the main monitor)
# ═══════════════════════════════════════════════════════════════════════
def range_axis(n):
    return np.arange(n // 2) * (SAMPLE_RATE_HZ / n) * C / (2 * SLOPE_HZ_PER_S)


def angle_axis(n_fft=AOA_FFT_SIZE):
    u = np.linspace(-1, 1, n_fft, endpoint=False)
    return np.degrees(np.arcsin(np.clip(u, -1, 1)))


def parse_one_frame(raw_bytes):
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    ret = np.zeros(len(raw) // 2, dtype=np.complex64)
    ret[0::2] = raw[0::4].astype(np.float32) + 1j * raw[2::4].astype(np.float32)
    ret[1::2] = raw[1::4].astype(np.float32) + 1j * raw[3::4].astype(np.float32)
    return ret.reshape(TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES)


def separate_tx(frame):
    tx1_elev = frame[0::3]
    tx0_azL  = frame[1::3]
    tx2_azR  = frame[2::3]
    return tx0_azL, tx1_elev, tx2_azR


def compute_range_fft(chirps, win):
    dc = chirps.mean(axis=-1, keepdims=True)
    windowed = (chirps - dc) * win
    rfft = np.fft.fft(windowed, axis=-1)[..., :NUM_ADC_SAMPLES // 2]
    return rfft.mean(axis=0)


def estimate_aoa(rfft_tx0, rfft_tx2, bin_idx, n_fft=AOA_FFT_SIZE):
    idx = min(bin_idx, rfft_tx0.shape[1] - 1)
    virtual = np.zeros(8, dtype=np.complex64)
    virtual[0:4] = rfft_tx0[:, idx]
    virtual[4:8] = rfft_tx2[:, idx]
    spectrum = np.fft.fftshift(np.fft.fft(virtual, n=n_fft))
    angles = angle_axis(n_fft)
    return float(angles[int(np.argmax(np.abs(spectrum)))])


def load_tx_calibration(path=CALIBRATION_FILE):
    if not os.path.exists(path):
        print(f"[CALIB] no calibration file '{path}' — AoA will be biased")
        return None
    try:
        d = np.load(path)
        corr = np.exp(1j * d["tx2_corrections"]).astype(np.complex64)
        print(f"[CALIB] loaded {path}")
        return corr
    except Exception as e:
        print(f"[CALIB] load error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# UDP capture
# ═══════════════════════════════════════════════════════════════════════
class UDPCapture(threading.Thread):
    def __init__(self, host, port, buf, lock):
        super().__init__(daemon=True)
        self.host, self.port, self.buf, self.lock = host, port, buf, lock
        self.running = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        sock.settimeout(1.0)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            print(f"ERROR: {e}"); self.running = False; return
        print(f"[UDP] listening on {self.host}:{self.port}")
        while self.running:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) > 10:
                with self.lock:
                    self.buf.extend(data[10:])
        sock.close()

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════
# Recorder — buffers everything to memory then dumps on stop
# ═══════════════════════════════════════════════════════════════════════
class Recorder:
    """Session recorder.

    We buffer to memory then write .npz on stop. This avoids I/O jitter
    during recording. For long sessions (>5 min in full mode), consider
    switching to streamed writes.
    """

    def __init__(self, out_dir=RECORDING_DIR):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.reset()

    def reset(self):
        self.recording = False
        self.mode = "summary"        # or "full"
        self.t_start = None
        self.frame_times = []        # wall time per frame (monotonic - epoch)
        self.raw_frames = []         # only in full mode
        self.range_profiles = []     # magnitude, bg-subtracted (1D array per frame)
        self.track_snapshots = []    # per-frame list of dicts
        self.session_name = None
        self.metadata = {}

    def start(self, mode, metadata):
        self.reset()
        self.recording = True
        self.mode = mode
        self.t_start = time.monotonic()
        self.metadata = metadata
        self.session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        print(f"[REC] START  mode={mode}  → {self.session_name}")

    def push(self, t_rel, raw_bytes, range_profile_mag, track_snapshot):
        """Buffer one frame's data."""
        if not self.recording:
            return
        self.frame_times.append(t_rel)
        self.range_profiles.append(range_profile_mag.astype(np.float32))
        self.track_snapshots.append(track_snapshot)
        if self.mode == "full":
            self.raw_frames.append(raw_bytes)

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        duration = time.monotonic() - self.t_start
        n_frames = len(self.frame_times)
        base = os.path.join(self.out_dir, self.session_name)

        print(f"[REC] STOP   {n_frames} frames  {duration:.1f}s → saving…")

        # ── 1. Track data → CSV ──────────────────────────────────────
        # One row per (frame, track) observation. Includes a frame_time
        # column for convenience (avoids join with frame_times array).
        csv_path = f"{base}_tracks.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame_idx",       # 0-based frame counter
                "frame_time_s",    # seconds since session start
                "track_id",
                "range_m",
                "angle_deg",
                "bpm",
                "validated",       # 1 if breathing pattern confirmed, else 0
                "breathing_score", # 0-1 spectral concentration
                "snr_db",          # spectral SNR of breathing peak
            ])
            for fi, snap in enumerate(self.track_snapshots):
                t = self.frame_times[fi] if fi < len(self.frame_times) else 0.0
                for trk in snap:
                    w.writerow([
                        fi,
                        f"{t:.4f}",
                        trk["track_id"],
                        f"{trk['range_m']:.3f}",
                        f"{trk['angle_deg']:.2f}",
                        f"{trk['bpm']:.2f}",
                        1 if trk["validated"] else 0,
                        f"{trk['breathing_score']:.4f}",
                        f"{trk['snr_db']:.2f}",
                    ])

        # ── 2. Metadata → JSON ───────────────────────────────────────
        meta_path = f"{base}_meta.json"
        meta_out = dict(self.metadata)   # copy
        meta_out["session"] = {
            "name": self.session_name,
            "duration_s": round(duration, 2),
            "n_frames": n_frames,
            "mode": self.mode,
            "files": {
                "tracks_csv": os.path.basename(csv_path),
            }
        }

        # ── 3. Signal data → NPZ (only in summary+ mode) ─────────────
        # Range profiles are always saved (small, high analysis value).
        # Raw ADC only saved in full mode.
        npz_path = f"{base}_signals.npz"
        save_dict = {
            "frame_times": np.array(self.frame_times, dtype=np.float64),
            "range_profiles": np.array(self.range_profiles, dtype=np.float32),
        }
        if self.mode == "full":
            raw_all = b"".join(self.raw_frames)
            save_dict["raw_adc"] = np.frombuffer(raw_all, dtype=np.int16).copy()
            save_dict["frame_layout"] = np.array(
                [TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES], dtype=np.int32)

        np.savez_compressed(npz_path, **save_dict)
        meta_out["session"]["files"]["signals_npz"] = os.path.basename(npz_path)

        with open(meta_path, "w") as f:
            json.dump(meta_out, f, indent=2)
        meta_out["session"]["files"]["metadata_json"] = os.path.basename(meta_path)

        # Sizes for feedback
        csv_mb   = os.path.getsize(csv_path)  / 1024 / 1024
        npz_mb   = os.path.getsize(npz_path)  / 1024 / 1024
        meta_kb  = os.path.getsize(meta_path) / 1024

        print(f"[REC] SAVED:")
        print(f"        tracks:    {csv_path}  ({csv_mb:.2f} MB)")
        print(f"        signals:   {npz_path}  ({npz_mb:.1f} MB)")
        print(f"        metadata:  {meta_path}  ({meta_kb:.1f} KB)")
        return base


# ═══════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Recording UI")
    ap.add_argument("--host", default=HOST_IP)
    ap.add_argument("--port", type=int, default=DATA_PORT)
    ap.add_argument("--out-dir", default=RECORDING_DIR)
    ap.add_argument("--notes", default="", help="metadata note stored with the recording")
    args = ap.parse_args()

    # Radar startup
    raw_buf = bytearray()
    buf_lock = threading.Lock()
    radar_src = UDPCapture(args.host, args.port, raw_buf, buf_lock)

    print("\n" + "═" * 55)
    print("  ENVIRONMENT CALIBRATION")
    print("  Ensure the monitored area is EMPTY of people.")
    print("═" * 55)
    input("  Press Enter when the area is clear… ")
    print()

    radar_src.start()
    t_epoch = time.monotonic()

    # Radar state
    rng = range_axis(NUM_ADC_SAMPLES)
    n_bins = NUM_ADC_SAMPLES // 2
    win_hann = np.hanning(NUM_ADC_SAMPLES).astype(np.float32)

    tx2_phase_correction = load_tx_calibration()

    search_min_bin = max(1, int(np.searchsorted(rng, SEARCH_MIN_M)))
    search_max_bin = min(n_bins - 1, int(np.searchsorted(rng, SEARCH_MAX_M)))

    bg_frames_buf = []
    bg_virtual_saved = None
    bg_noise_std = None
    frame_count = 0
    track_mgr = None
    track_angles = {}

    recorder = Recorder(out_dir=args.out_dir)

    # ── UI ──
    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.canvas.manager.set_window_title("Breathing Recorder")

    # Layout: big status area on top, buttons on bottom
    ax_status = fig.add_axes([0.05, 0.35, 0.90, 0.60])
    ax_status.set_facecolor("#16213e")
    ax_status.set_xticks([]); ax_status.set_yticks([])
    for sp in ax_status.spines.values():
        sp.set_color("#333")

    # Header
    txt_header = ax_status.text(
        0.5, 0.92, "calibrating…",
        transform=ax_status.transAxes, ha="center", va="top",
        color="#ff9f43", fontsize=13, fontweight="bold")

    # Recording indicator
    txt_rec_status = ax_status.text(
        0.5, 0.82, "not recording",
        transform=ax_status.transAxes, ha="center", va="top",
        color="#888888", fontsize=11)

    # Track slots
    slot_texts = []  # (label, bpm) pairs
    for i in range(MAX_TRACKS):
        y = 0.62 - i * 0.14
        color = TRACK_COLORS[i]
        label = ax_status.text(
            0.05, y, f"track {i+1}",
            transform=ax_status.transAxes, va="center",
            color="#888", fontsize=10)
        info = ax_status.text(
            0.30, y, "—",
            transform=ax_status.transAxes, va="center",
            color="#666", fontsize=11, family="monospace")
        bpm = ax_status.text(
            0.75, y, "--",
            transform=ax_status.transAxes, va="center",
            color="#333", fontsize=22, family="monospace", fontweight="bold")
        slot_texts.append((label, info, bpm))

    # Session duration
    txt_duration = ax_status.text(
        0.05, 0.02, "",
        transform=ax_status.transAxes, ha="left", va="bottom",
        color="#666", fontsize=9)

    # File size estimate during recording
    txt_size = ax_status.text(
        0.95, 0.02, "",
        transform=ax_status.transAxes, ha="right", va="bottom",
        color="#666", fontsize=9)

    # ── Buttons ──
    ax_mode = fig.add_axes([0.05, 0.05, 0.20, 0.18])
    ax_mode.set_facecolor("#16213e")
    radio_mode = RadioButtons(
        ax_mode, ("summary", "full"),
        active=0, activecolor="#00ff99")
    for lbl in radio_mode.labels:
        lbl.set_color("white")
        lbl.set_fontsize(10)

    ax_start = fig.add_axes([0.35, 0.10, 0.25, 0.08])
    btn_start = Button(ax_start, "START RECORDING",
                       color="#00b050", hovercolor="#00d060")
    btn_start.label.set_color("white")
    btn_start.label.set_fontweight("bold")

    ax_stop = fig.add_axes([0.65, 0.10, 0.25, 0.08])
    btn_stop = Button(ax_stop, "STOP RECORDING",
                       color="#333", hovercolor="#555")
    btn_stop.label.set_color("#888")

    def on_start(_):
        if recorder.recording:
            return
        mode = radio_mode.value_selected
        metadata = {
            "notes": args.notes,
            "start_time": datetime.now().isoformat(),
            "radar": {
                "num_tx": NUM_TX, "num_rx": NUM_RX,
                "num_chirps": NUM_CHIRPS, "num_adc_samples": NUM_ADC_SAMPLES,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "slope_hz_per_s": SLOPE_HZ_PER_S,
                "fps": FPS_HZ,
                "calibration_loaded": tx2_phase_correction is not None,
            },
        }
        recorder.start(mode, metadata)
        btn_start.ax.set_facecolor("#333")
        btn_start.label.set_color("#888")
        btn_stop.ax.set_facecolor("#c53030")
        btn_stop.label.set_color("white")
        txt_rec_status.set_color("#ff4040")

    def on_stop(_):
        if not recorder.recording:
            return
        base = recorder.stop()
        btn_start.ax.set_facecolor("#00b050")
        btn_start.label.set_color("white")
        btn_stop.ax.set_facecolor("#333")
        btn_stop.label.set_color("#888")
        txt_rec_status.set_text(
            f"not recording — saved as\n{os.path.basename(base)}_tracks.csv (+meta.json, +signals.npz)")
        txt_rec_status.set_color("#00ff99")

    btn_start.on_clicked(on_start)
    btn_stop.on_clicked(on_stop)

    # ── Processing loop (via animation) ──
    def update(_):
        nonlocal frame_count, bg_virtual_saved, bg_noise_std, track_mgr

        # Snapshot for recording (built during processing)
        this_frame_snapshot = None
        this_range_profile = None
        this_raw = None

        with buf_lock:
            frames_ready = []
            while len(raw_buf) >= RAW_BYTES_PER_FRAME:
                chunk = bytes(raw_buf[:RAW_BYTES_PER_FRAME])
                del raw_buf[:RAW_BYTES_PER_FRAME]
                frames_ready.append(chunk)

        for raw_chunk in frames_ready:
            frame = parse_one_frame(raw_chunk)
            wall_now = time.monotonic()

            tx0_chirps, tx1_chirps, tx2_chirps = separate_tx(frame)
            rfft_tx0 = compute_range_fft(tx0_chirps, win_hann)
            rfft_tx1 = compute_range_fft(tx1_chirps, win_hann)
            rfft_tx2 = compute_range_fft(tx2_chirps, win_hann)
            if tx2_phase_correction is not None:
                rfft_tx2 = rfft_tx2 * tx2_phase_correction[:, np.newaxis]

            virtual = np.concatenate([rfft_tx0, rfft_tx1, rfft_tx2], axis=0)

            frame_count += 1
            t_rel = wall_now - t_epoch

            # Background calibration
            if frame_count <= BG_FRAMES:
                bg_frames_buf.append(virtual.copy())
                pct = frame_count / BG_FRAMES * 100
                secs_left = (BG_FRAMES - frame_count) * FRAME_PERIOD_S
                txt_header.set_text(f"calibrating background… {pct:.0f}%  ({secs_left:.0f}s)")
                continue

            if bg_virtual_saved is None:
                all_v = np.array(bg_frames_buf)
                bg_virtual_saved = np.mean(all_v, axis=0)
                avg_all = all_v.mean(axis=1)
                bg_noise_std = np.std(np.abs(avg_all), axis=0)
                bg_frames_buf.clear()
                print("[RADAR] background calibration complete")

            diff_virtual = virtual - bg_virtual_saved
            diff_avg = diff_virtual.mean(axis=0)
            mag_det = np.abs(diff_avg)

            # Raw phase extraction for tracking
            azimuth_raw = np.concatenate([rfft_tx0, rfft_tx2], axis=0)
            rfft_avg_raw = azimuth_raw.mean(axis=0)

            if track_mgr is None:
                track_mgr = MultiTrackManager(
                    range_axis=rng, lambda_m=LAMBDA_M, fps=FPS_HZ,
                    min_bin=search_min_bin, max_bin=search_max_bin,
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
                if bg_noise_std is not None:
                    track_mgr.set_background_stats(bg_noise_std, cfar_k=CFAR_K)

            track_mgr.step(mag_det, rfft_avg_raw, t_rel)

            for trk in track_mgr.confirmed_tracks:
                bi = int(round(trk.bin_position))
                track_angles[trk.track_id] = estimate_aoa(rfft_tx0, rfft_tx2, bi)

            # Build per-track snapshot for recording
            snapshot = []
            for trk in track_mgr.confirmed_tracks:
                snapshot.append({
                    "track_id": int(trk.track_id),
                    "range_m": float(trk.range_m),
                    "angle_deg": float(track_angles.get(trk.track_id, 0.0)),
                    "bpm": float(trk.bpm),
                    "validated": bool(trk.breathing_validated),
                    "breathing_score": float(getattr(trk, "breathing_score", 0.0)),
                    "snr_db": float(getattr(trk, "peak_snr_db", 0.0)),
                })

            this_frame_snapshot = snapshot
            this_range_profile = mag_det
            this_raw = raw_chunk

            # Feed the recorder
            recorder.push(t_rel, this_raw, this_range_profile, snapshot)

        # ── Update UI ──
        if track_mgr is not None and bg_virtual_saved is not None:
            confirmed = track_mgr.confirmed_tracks
            n_conf = len(confirmed)
            n_val = sum(1 for t in confirmed if t.breathing_validated)

            txt_header.set_text(f"{n_val} breathing  ·  {n_conf - n_val} pending")
            txt_header.set_color("#00ff99" if n_val > 0 else "#ff9f43")

            for i, (label, info, bpm) in enumerate(slot_texts):
                if i < n_conf:
                    t = confirmed[i]
                    a = track_angles.get(t.track_id, 0)
                    label.set_color(t.color)
                    if t.breathing_validated:
                        info.set_text(f"{t.range_m:4.1f} m   {a:+4.0f}°")
                        info.set_color(t.color)
                        bpm.set_text(f"{t.bpm:.0f}" if t.bpm > 0 else "--")
                        bpm.set_color(t.color)
                    else:
                        info.set_text(f"{t.range_m:4.1f} m   {a:+4.0f}°  (pending)")
                        info.set_color("#888")
                        bpm.set_text("…")
                        bpm.set_color("#555")
                else:
                    label.set_color("#444")
                    info.set_text("—")
                    info.set_color("#333")
                    bpm.set_text("--")
                    bpm.set_color("#333")

        # Recording status
        if recorder.recording:
            elapsed = time.monotonic() - recorder.t_start
            n = len(recorder.frame_times)
            mins, secs = divmod(int(elapsed), 60)
            txt_rec_status.set_text(f"● RECORDING  {mins:02d}:{secs:02d}  ({recorder.mode})")
            txt_duration.set_text(f"frames: {n}")

            # Estimate size
            if recorder.mode == "full":
                mb = n * RAW_BYTES_PER_FRAME / 1024 / 1024
                txt_size.set_text(f"est. {mb:.0f} MB")
            else:
                # summary: range profile (n_bins × 4 bytes) + small overhead per frame
                mb = n * n_bins * 4 / 1024 / 1024
                txt_size.set_text(f"est. {mb:.1f} MB")
        else:
            txt_duration.set_text("")
            txt_size.set_text("")

    ani = FuncAnimation(fig, update, interval=100, blit=False,
                        cache_frame_data=False)

    print("\n[MAIN] recorder UI running")
    print("[MAIN]   summary mode = ~1 MB/minute (track state + range profiles)")
    print("[MAIN]   full mode    = ~2.3 GB/minute (adds raw ADC frames)")
    print()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    if recorder.recording:
        print("\n[MAIN] recording still active — stopping and saving…")
        recorder.stop()

    radar_src.stop()
    radar_src.join(timeout=2)
    print("[DONE]")


if __name__ == "__main__":
    main()