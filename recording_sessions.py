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
  python recording_sessions.py
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
from matplotlib.widgets import Button, RadioButtons, TextBox

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
BELT_NOMINAL_HZ   = 10
BELT_PERIOD_MS    = 100

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
# Belt ground truth (mirrors radar_vitals.py)
# ═══════════════════════════════════════════════════════════════════════
def ask_belt():
    print("\n  Respiration belt ground truth (optional):")
    try:
        ans = input("  Connect GDX belt for this recording? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    return ans == "y"


def select_belt_device():
    try:
        from godirect import GoDirect
    except ImportError:
        print("ERROR: godirect not installed.  pip install godirect")
        return None, None

    godirect = GoDirect(use_ble=True, use_usb=False)
    print("\n[BLE] scanning for Go Direct devices …")
    devices = godirect.list_devices()
    gdx = [d for d in devices if d.name and d.name.upper().startswith("GDX")]
    if not gdx:
        print("  No GDX devices found. Continuing without belt.")
        godirect.quit()
        return None, None

    print(f"\n  Found {len(gdx)} GDX device(s):\n")
    for i, d in enumerate(gdx):
        print(f"    [{i}]  {d.name}   (order: {d.order_code})")
    if len(gdx) == 1:
        choice = 0
        print(f"\n  Auto-selecting [{choice}] {gdx[choice].name}")
    else:
        while True:
            try:
                choice = int(input(f"\n  Select [0-{len(gdx)-1}]: ").strip())
                if 0 <= choice < len(gdx):
                    break
            except (ValueError, EOFError):
                pass
    return godirect, gdx[choice]


class BeltReader(threading.Thread):
    def __init__(self, device, data_deque, lock, period_ms=BELT_PERIOD_MS):
        super().__init__(daemon=True)
        self.device, self.data, self.lock = device, data_deque, lock
        self.period_ms = period_ms
        self.running = True
        self.t0 = None
        self.sensor_description = ""
        self.sensor_units = ""

    def run(self):
        dev = self.device
        if not dev.open():
            print("[BELT] ERROR: could not open device")
            self.running = False
            return
        dev.enable_default_sensors()
        enabled = dev.get_enabled_sensors()
        force_sensor = None
        for s in enabled:
            if "force" in s.sensor_description.lower():
                force_sensor = s
                break
        if force_sensor is None and enabled:
            force_sensor = enabled[0]
        if force_sensor is None:
            print("[BELT] ERROR: no enabled sensors")
            self.running = False
            return

        self.sensor_description = force_sensor.sensor_description
        self.sensor_units = force_sensor.sensor_units
        print(f"[BELT] streaming {self.sensor_description} "
              f"({self.sensor_units}) @ {1000/self.period_ms:.0f} Hz")

        dev.start(period=self.period_ms)
        self.t0 = time.monotonic()
        while self.running:
            try:
                if dev.read():
                    v = force_sensor.value
                    if v is not None and v == v:
                        with self.lock:
                            self.data.append((time.monotonic(), float(v)))
            except Exception as e:
                if self.running:
                    print(f"[BELT] error: {e}")
                break
        try:
            dev.stop()
            dev.close()
        except Exception:
            pass

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
        # Complex range FFTs per virtual element BEFORE bg subtraction and
        # angle FFT — lets replay reconstruct the fan and change AoA later.
        # Shape per frame: (12, n_bins) complex64
        self.range_complex_frames = []
        self.track_snapshots = []    # per-frame list of dicts
        self.belt_times = []         # seconds relative to shared t_epoch
        self.belt_force = []         # raw belt Force channel samples
        self.session_name = None
        self.metadata = {}

    @staticmethod
    def sanitize_description(desc, max_len=20):
        """Turn free-form description into a filename-safe suffix.

        Rules:
          - Lowercase, spaces → underscores
          - Drop anything that isn't alphanumeric, underscore, or dash
          - Collapse multiple underscores
          - Trim to max_len characters
          - Empty result yields empty string (no dangling underscore)
        """
        if not desc:
            return ""
        s = desc.strip().lower().replace(" ", "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch in "_-")
        # Collapse consecutive underscores
        while "__" in s:
            s = s.replace("__", "_")
        s = s.strip("_-")
        return s[:max_len]

    def start(self, mode, metadata, description=""):
        self.reset()
        self.recording = True
        self.mode = mode
        self.t_start = time.monotonic()
        self.metadata = dict(metadata)
        self.metadata["description"] = description

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        desc_slug = self.sanitize_description(description, max_len=20)
        if desc_slug:
            self.session_name = f"session_{stamp}_{desc_slug}"
        else:
            self.session_name = f"session_{stamp}"
        print(f"[REC] START  mode={mode}  → {self.session_name}")

    def push(self, t_rel, raw_bytes, range_profile_mag, range_complex, track_snapshot):
        """Buffer one frame's data.

        Args:
            t_rel: seconds since session start
            raw_bytes: raw ADC bytes (used only in full mode)
            range_profile_mag: (n_bins,) float magnitude, bg-subtracted
            range_complex: (12, n_bins) complex64 range FFT per virtual element,
                           BEFORE bg subtraction. Enables offline replay.
            track_snapshot: list of per-track dicts for this frame
        """
        if not self.recording:
            return
        self.frame_times.append(t_rel)
        self.range_profiles.append(range_profile_mag.astype(np.float32))
        self.range_complex_frames.append(range_complex.astype(np.complex64))
        self.track_snapshots.append(track_snapshot)
        if self.mode == "full":
            self.raw_frames.append(raw_bytes)

    def set_belt_samples(self, samples, t_epoch, stop_wall=None):
        if self.t_start is None:
            return
        stop_wall = time.monotonic() if stop_wall is None else stop_wall
        filtered = [(wall, val) for wall, val in samples
                    if self.t_start <= wall <= stop_wall]
        self.belt_times = [wall - t_epoch for wall, _ in filtered]
        self.belt_force = [val for _, val in filtered]

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        duration = time.monotonic() - self.t_start
        n_frames = len(self.frame_times)

        # Create a folder per session; files inside share the session name.
        session_dir = os.path.join(self.out_dir, self.session_name)
        os.makedirs(session_dir, exist_ok=True)
        base = os.path.join(session_dir, self.session_name)

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
        if "belt" in meta_out:
            meta_out["belt"]["n_samples_recorded"] = len(self.belt_force)

        # ── 3. Signal data → NPZ (always saved) ──────────────────────
        # Range profiles + complex range FFTs are always saved. Raw ADC
        # only saved in full mode.
        npz_path = f"{base}_signals.npz"
        save_dict = {
            "frame_times": np.array(self.frame_times, dtype=np.float64),
            "range_profiles": np.array(self.range_profiles, dtype=np.float32),
            # (n_frames, 12, n_bins) complex64. This is the key data for
            # offline replay — everything downstream of the range FFT can
            # be recomputed from it.
            "range_complex": np.array(self.range_complex_frames, dtype=np.complex64),
        }
        if self.metadata.get("belt", {}).get("enabled") or self.belt_force:
            save_dict["belt_times"] = np.array(self.belt_times, dtype=np.float64)
            save_dict["belt_force"] = np.array(self.belt_force, dtype=np.float32)
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
    ap.add_argument("--belt-period", type=int, default=BELT_PERIOD_MS,
                    help="Go Direct belt sampling period in ms")
    args = ap.parse_args()

    # Radar startup
    raw_buf = bytearray()
    buf_lock = threading.Lock()
    radar_src = UDPCapture(args.host, args.port, raw_buf, buf_lock)

    # Optional belt ground truth
    use_belt = False
    godirect_ctx = None
    belt_reader = None
    belt_data = collections.deque()
    belt_lock = threading.Lock()
    if ask_belt():
        godirect_ctx, belt_device = select_belt_device()
        if godirect_ctx and belt_device:
            belt_reader = BeltReader(belt_device, belt_data, belt_lock,
                                     period_ms=args.belt_period)
            use_belt = True
        else:
            print("  Proceeding without belt.")

    print("\n" + "═" * 55)
    print("  ENVIRONMENT CALIBRATION")
    print("  Ensure the monitored area is EMPTY of people.")
    print("═" * 55)
    input("  Press Enter when the area is clear… ")
    print()

    if use_belt:
        belt_reader.start()
    radar_src.start()

    if use_belt:
        print("[MAIN] waiting for belt …")
        deadline = time.monotonic() + 15
        while belt_reader.t0 is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if belt_reader.t0 is None:
            print("WARNING: belt did not produce data within 15 s")
    t_epoch = belt_reader.t0 if use_belt and belt_reader.t0 else time.monotonic()

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

    # ── Description text box ──
    # Positioned at the bottom, above the mode/buttons row.
    # Max 20 chars used from this string when forming the folder name.
    ax_desc_label = fig.add_axes([0.05, 0.28, 0.20, 0.03])
    ax_desc_label.set_facecolor("#1a1a2e")
    ax_desc_label.set_xticks([]); ax_desc_label.set_yticks([])
    for sp in ax_desc_label.spines.values():
        sp.set_visible(False)
    ax_desc_label.text(0.0, 0.5, "description (max 20 chars):",
                       transform=ax_desc_label.transAxes, va="center",
                       color="#aaa", fontsize=10)

    ax_desc = fig.add_axes([0.27, 0.28, 0.65, 0.04])
    ax_desc.set_facecolor("#16213e")
    txtbox_desc = TextBox(ax_desc, "", initial="",
                          color="#16213e", hovercolor="#1e2a4a",
                          textalignment="left")
    txtbox_desc.text_disp.set_color("white")
    txtbox_desc.text_disp.set_fontsize(10)
    for sp in ax_desc.spines.values():
        sp.set_color("#333")

    # Live preview of what folder name will be generated
    txt_preview = fig.text(0.27, 0.245,
                           "→ session_YYYYMMDD_HHMMSS",
                           color="#666", fontsize=8)

    def on_desc_change(text):
        slug = Recorder.sanitize_description(text, max_len=20)
        stamp = "YYYYMMDD_HHMMSS"
        if slug:
            txt_preview.set_text(f"→ session_{stamp}_{slug}")
        else:
            txt_preview.set_text(f"→ session_{stamp}")
        fig.canvas.draw_idle()

    txtbox_desc.on_text_change(on_desc_change)

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
        description = txtbox_desc.text.strip()
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
            "belt": {
                "enabled": use_belt,
                "period_ms": args.belt_period if use_belt else None,
                "nominal_hz": BELT_NOMINAL_HZ if use_belt else None,
                "sensor": belt_reader.sensor_description if use_belt else "",
                "units": belt_reader.sensor_units if use_belt else "",
            },
        }
        recorder.start(mode, metadata, description=description)
        btn_start.ax.set_facecolor("#333")
        btn_start.label.set_color("#888")
        btn_stop.ax.set_facecolor("#c53030")
        btn_stop.label.set_color("white")
        txt_rec_status.set_color("#ff4040")

    def on_stop(_):
        if not recorder.recording:
            return
        if use_belt:
            stop_wall = time.monotonic()
            with belt_lock:
                belt_snapshot = list(belt_data)
            recorder.set_belt_samples(belt_snapshot, t_epoch, stop_wall=stop_wall)
        base = recorder.stop()
        btn_start.ax.set_facecolor("#00b050")
        btn_start.label.set_color("white")
        btn_stop.ax.set_facecolor("#333")
        btn_stop.label.set_color("#888")
        session_name = os.path.basename(base)
        txt_rec_status.set_text(
            f"saved to {session_name}/  (tracks.csv, meta.json, signals.npz)")
        txt_rec_status.set_color("#00ff99")
        # Clear the description textbox so it's ready for the next session
        txtbox_desc.set_val("")

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

            # Feed the recorder — save the pre-bg-subtraction complex range
            # FFTs per virtual element so offline replay can vary bg subtraction
            # and AoA settings independently.
            recorder.push(t_rel, this_raw, this_range_profile, virtual, snapshot)

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
                # summary now includes complex range FFTs (12 elem × n_bins × 8 bytes)
                # plus 1D range profile (n_bins × 4 bytes)
                per_frame = 12 * n_bins * 8 + n_bins * 4
                mb = n * per_frame / 1024 / 1024
                txt_size.set_text(f"est. {mb:.1f} MB")
        else:
            txt_duration.set_text("")
            txt_size.set_text("")

    ani = FuncAnimation(fig, update, interval=100, blit=False,
                        cache_frame_data=False)

    print("\n[MAIN] recorder UI running")
    print("[MAIN]   summary mode = ~40 MB/minute (tracks + complex range FFTs)")
    print("[MAIN]                    ↳ enough to replay offline with different algorithms")
    print("[MAIN]   full mode    = ~2.3 GB/minute (adds raw ADC frames)")
    print("[MAIN]                    ↳ enough to also change range-FFT parameters")
    print()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    if recorder.recording:
        print("\n[MAIN] recording still active — stopping and saving…")
        if use_belt:
            stop_wall = time.monotonic()
            with belt_lock:
                belt_snapshot = list(belt_data)
            recorder.set_belt_samples(belt_snapshot, t_epoch, stop_wall=stop_wall)
        recorder.stop()

    if belt_reader:
        belt_reader.stop()
        belt_reader.join(timeout=3)
    radar_src.stop()
    radar_src.join(timeout=2)
    if godirect_ctx:
        godirect_ctx.quit()
    print("[DONE]")


if __name__ == "__main__":
    main()
