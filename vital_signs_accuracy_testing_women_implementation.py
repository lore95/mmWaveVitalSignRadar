#!/usr/bin/env python3
"""
Dual-source real-time breathing monitor
────────────────────────────────────────
Combines:
  • TI mmWave radar + DCA1000  (UDP)
  • Vernier Go Direct Respiration Belt  (BLE)

Both streams are timestamped with time.monotonic() at the moment each
sample is received on the PC, giving a shared wall-clock reference.

Layout:
  ┌──────────────────┬──────────┐
  │  range profile   │ radar BPM│
  │  (radar)         │ belt BPM │
  ├──────────────────┴──────────┤
  │  radar chest displacement   │
  ├─────────────────────────────┤
  │  belt force waveform        │
  └─────────────────────────────┘

Usage:
  python dual_breathing_monitor.py --demo          # synthetic radar, real belt
  python dual_breathing_monitor.py                  # real radar + real belt
  python dual_breathing_monitor.py --belt-only      # belt only, no radar
"""

import argparse
import threading
import socket
import struct
import time
import sys
import collections

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt

# ═══════════════════════════════════════════════════════════════════════
# Radar / capture constants  (match your .json / DCA1000 config)
# ═══════════════════════════════════════════════════════════════════════
NUM_RX            = 1
NUM_TX            = 1
NUM_ADC_SAMPLES   = 540
NUM_CHIRPS        = 128
FRAME_PERIOD_S    = 39.99975e-3          # ~25 Hz
FPS_HZ            = 1.0 / FRAME_PERIOD_S

C                 = 2.998e8
SLOPE_HZ_PER_S   = 70.006e12
SAMPLE_RATE_HZ    = 10e6
F0_HZ             = 77.0000000238419e9
LAMBDA_M          = C / F0_HZ

COMPLEX_PER_FRAME = NUM_CHIRPS * NUM_TX * NUM_RX * NUM_ADC_SAMPLES
RAW_BYTES_PER_FRAME = COMPLEX_PER_FRAME * 8

BG_FRAMES         = 50
DETECT_FRAMES     = 125        # ~5 s at 25 Hz — breathing detection window
HISTORY_S         = 30
MAX_HISTORY       = int(HISTORY_S * FPS_HZ)
BP_LO, BP_HI     = 0.1, 0.6   # breathing band (Hz)
BPM_WINDOW_S      = 15
XMAX_M            = 2.0

HOST_IP           = "192.168.33.30"
DATA_PORT         = 4098

# Belt constants
BELT_HISTORY_S    = 30
BELT_NOMINAL_HZ   = 10         # ~10 Hz from the dump data
MAX_BELT_HISTORY  = int(BELT_HISTORY_S * BELT_NOMINAL_HZ)
BELT_BPM_WINDOW_S = 15
BELT_BP_LO        = 0.1
BELT_BP_HI        = 0.6


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def range_axis(n: int) -> np.ndarray:
    return np.arange(n // 2) * (SAMPLE_RATE_HZ / n) * C / (2 * SLOPE_HZ_PER_S)


def make_bandpass(lo, hi, fs, order=4):
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def estimate_bpm(signal_seg, fs):
    """Spectral peak in the breathing band → BPM."""
    if len(signal_seg) < 64:
        return 0.0, 0.0
    sig = signal_seg - signal_seg.mean()
    f = np.fft.rfftfreq(len(sig), d=1 / fs)
    mag = np.abs(np.fft.rfft(sig))
    band = (f >= BP_LO) & (f <= BP_HI)
    if not band.any():
        return 0.0, 0.0
    peak_f = float(f[band][np.argmax(mag[band])])
    return peak_f * 60.0, peak_f


def find_breathing_bin(phase_matrix, fs, range_axis_m,
                       min_range=0.2, max_range=1.5,
                       min_mag=None, mag_profile=None):
    """Score each range bin by breathing-band phase energy.

    Args:
        phase_matrix: (n_frames, n_bins) unwrapped phase per bin per frame
        fs: frame rate (Hz)
        range_axis_m: range in metres per bin
        min_range, max_range: search window (m)
        mag_profile: optional (n_bins,) magnitude — used to exclude
                     very weak bins (noise) from consideration

    Returns:
        best_bin (int), scores (np.ndarray of shape n_bins)
    """
    n_frames, n_bins = phase_matrix.shape
    scores = np.zeros(n_bins)

    lo_bin = max(1, int(np.searchsorted(range_axis_m, min_range)))
    hi_bin = min(n_bins, int(np.searchsorted(range_axis_m, max_range)))

    # Optional: only consider bins with reasonable magnitude
    if mag_profile is not None:
        noise_floor = np.median(mag_profile[lo_bin:hi_bin])
        mag_mask = mag_profile > noise_floor * 2
    else:
        mag_mask = np.ones(n_bins, dtype=bool)

    for b in range(lo_bin, hi_bin):
        if not mag_mask[b]:
            continue
        ph = phase_matrix[:, b]
        # detrend (remove slow drift)
        ph = ph - np.polyval(np.polyfit(np.arange(len(ph)), ph, 1),
                             np.arange(len(ph)))
        # spectral energy in breathing band
        f = np.fft.rfftfreq(len(ph), d=1 / fs)
        mag = np.abs(np.fft.rfft(ph))
        band = (f >= BP_LO) & (f <= BP_HI)
        if band.any():
            scores[b] = float(np.sum(mag[band] ** 2))

    best = lo_bin + int(np.argmax(scores[lo_bin:hi_bin]))
    return best, scores


# ═══════════════════════════════════════════════════════════════════════
# BLE device selector  (runs before any threads start)
# ═══════════════════════════════════════════════════════════════════════

def select_belt_device():
    """Scan for Go Direct BLE devices whose name starts with GDX,
    present a numbered list, return the selected device object."""
    try:
        from godirect import GoDirect
    except ImportError:
        print("ERROR: godirect not installed.  pip install godirect")
        sys.exit(1)

    godirect = GoDirect(use_ble=True, use_usb=False)
    print("\n[BLE] scanning for Go Direct devices …")
    devices = godirect.list_devices()

    gdx_devices = [d for d in devices if d.name and d.name.upper().startswith("GDX")]

    if not gdx_devices:
        print("  No GDX devices found. Is the belt powered on?")
        godirect.quit()
        sys.exit(1)

    print(f"\n  Found {len(gdx_devices)} GDX device(s):\n")
    for i, d in enumerate(gdx_devices):
        print(f"    [{i}]  {d.name}   (order: {d.order_code})")

    if len(gdx_devices) == 1:
        choice = 0
        print(f"\n  Auto-selecting [{choice}] {gdx_devices[choice].name}")
    else:
        while True:
            try:
                raw = input(f"\n  Select device [0-{len(gdx_devices)-1}]: ").strip()
                choice = int(raw)
                if 0 <= choice < len(gdx_devices):
                    break
            except (ValueError, EOFError):
                pass
            print("  Invalid selection, try again.")

    return godirect, gdx_devices[choice]


# ═══════════════════════════════════════════════════════════════════════
# Belt reader thread
# ═══════════════════════════════════════════════════════════════════════

class BeltReader(threading.Thread):
    """Reads Force channel from a Go Direct device and pushes
    (wall_time, force_N) into a shared deque."""

    def __init__(self, device, data_deque, lock, period_ms=100):
        super().__init__(daemon=True)
        self.device = device
        self.data = data_deque        # deque of (float, float)
        self.lock = lock
        self.period_ms = period_ms
        self.running = True
        self.t0 = None                # set at first sample

    def run(self):
        dev = self.device

        if not dev.open():
            print("[BELT] ERROR: could not open device")
            self.running = False
            return

        # Enable only the Force channel (channel 1 on GDX-RB)
        dev.enable_default_sensors()
        enabled = dev.get_enabled_sensors()
        # Find the Force sensor specifically
        force_sensor = None
        for s in enabled:
            if "force" in s.sensor_description.lower():
                force_sensor = s
                break
        if force_sensor is None and enabled:
            force_sensor = enabled[0]  # fallback to first

        print(f"[BELT] connected — streaming {force_sensor.sensor_description} "
              f"({force_sensor.sensor_units}) @ {1000/self.period_ms:.0f} Hz")

        dev.start(period=self.period_ms)
        self.t0 = time.monotonic()

        while self.running:
            try:
                if dev.read():
                    wall = time.monotonic()
                    val = force_sensor.value
                    if val is not None and val == val:  # skip NaN
                        with self.lock:
                            self.data.append((wall, float(val)))
            except Exception as e:
                if self.running:
                    print(f"[BELT] read error: {e}")
                break

        try:
            dev.stop()
            dev.close()
        except Exception:
            pass
        print("[BELT] stopped")

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════
# UDP capture thread  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════

class UDPCapture(threading.Thread):
    """Listens for DCA1000 UDP packets and pushes raw ADC bytes into a buffer."""

    def __init__(self, host, port, buf: bytearray, lock: threading.Lock):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.buf = buf
        self.lock = lock
        self.running = True
        self.total_bytes = 0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.settimeout(1.0)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            print(f"ERROR: Cannot bind {self.host}:{self.port} — {e}")
            self.running = False
            return
        print(f"[UDP] listening on {self.host}:{self.port}")

        while self.running:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) > 10:
                payload = data[10:]
                with self.lock:
                    self.buf.extend(payload)
                self.total_bytes += len(payload)
        sock.close()

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════
# Demo source  (synthetic radar, unchanged)
# ═══════════════════════════════════════════════════════════════════════

class DemoSource(threading.Thread):
    """Generates fake radar frames at ~25 Hz for UI testing."""

    def __init__(self, buf: bytearray, lock: threading.Lock, bpm=15.0):
        super().__init__(daemon=True)
        self.buf = buf
        self.lock = lock
        self.running = True
        self.bpm = bpm
        self.frame_idx = 0

    def run(self):
        print(f"[DEMO] synthetic radar @ {FPS_HZ:.0f} Hz, breathing = {self.bpm:.0f} BPM")
        rng = range_axis(NUM_ADC_SAMPLES)
        target_bin = int(np.searchsorted(rng, 0.5))
        breath_f = self.bpm / 60.0

        while self.running:
            t = self.frame_idx * FRAME_PERIOD_S
            frame_iq = np.zeros((NUM_CHIRPS, NUM_RX, NUM_ADC_SAMPLES), dtype=np.complex64)
            phase_shift = 0.005 * np.sin(2 * np.pi * breath_f * t)
            freq_bin = target_bin / NUM_ADC_SAMPLES
            n = np.arange(NUM_ADC_SAMPLES)
            for c in range(NUM_CHIRPS):
                tone = 1000 * np.exp(1j * (2 * np.pi * freq_bin * n + phase_shift))
                tone += (np.random.randn(NUM_ADC_SAMPLES) +
                         1j * np.random.randn(NUM_ADC_SAMPLES)) * 5
                frame_iq[c, 0, :] = tone

            flat = frame_iq.reshape(-1)
            i_vals = np.round(flat.real).astype(np.int16)
            q_vals = np.round(flat.imag).astype(np.int16)
            raw = np.empty(len(flat) * 4, dtype=np.int16)
            raw[0::4] = i_vals
            raw[1::4] = 0
            raw[2::4] = q_vals
            raw[3::4] = 0

            with self.lock:
                self.buf.extend(raw.tobytes())
            self.frame_idx += 1
            time.sleep(FRAME_PERIOD_S)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════
# Frame parser
# ═══════════════════════════════════════════════════════════════════════

def parse_one_frame(raw_bytes: bytes) -> np.ndarray:
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    samples = raw[0::2]
    iq = samples[0::2].astype(np.float32) + 1j * samples[1::2].astype(np.float32)
    return iq.reshape(NUM_CHIRPS * NUM_TX, NUM_RX, NUM_ADC_SAMPLES)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Dual-source breathing monitor")
    ap.add_argument("--host", default=HOST_IP)
    ap.add_argument("--port", type=int, default=DATA_PORT)
    ap.add_argument("--demo", action="store_true",
                    help="synthetic radar data (no radar hardware)")
    ap.add_argument("--bpm", type=float, default=15.0,
                    help="simulated breathing rate for --demo")
    ap.add_argument("--belt-only", action="store_true",
                    help="run belt only, no radar source")
    ap.add_argument("--belt-period", type=int, default=100,
                    help="belt sampling period in ms (default 100 = 10 Hz)")
    args = ap.parse_args()

    # ── 1. Select and prepare the belt ─────────────────────────────
    godirect_ctx, belt_device = select_belt_device()

    belt_data = collections.deque(maxlen=MAX_BELT_HISTORY)
    belt_lock = threading.Lock()

    belt_reader = BeltReader(belt_device, belt_data, belt_lock,
                             period_ms=args.belt_period)

    # ── 2. Prepare radar source ────────────────────────────────────
    radar_enabled = not args.belt_only
    raw_buf = bytearray()
    buf_lock = threading.Lock()
    radar_src = None

    if radar_enabled:
        if args.demo:
            radar_src = DemoSource(raw_buf, buf_lock, bpm=args.bpm)
        else:
            radar_src = UDPCapture(args.host, args.port, raw_buf, buf_lock)

    # ── 3. Start data threads ──────────────────────────────────────
    belt_reader.start()
    if radar_src:
        radar_src.start()

    # Wait until belt has its t0 set (i.e. first sample arrived)
    print("[MAIN] waiting for belt first sample …")
    deadline = time.monotonic() + 15
    while belt_reader.t0 is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if belt_reader.t0 is None:
        print("WARNING: belt did not produce data within 15 s")

    # Use a shared epoch so both clocks show seconds-since-start
    t_epoch = belt_reader.t0 if belt_reader.t0 else time.monotonic()

    # ── 4. Radar processing state ──────────────────────────────────
    rng = range_axis(NUM_ADC_SAMPLES)
    n_bins = NUM_ADC_SAMPLES // 2
    keep_mask = rng <= XMAX_M
    rng_plot = rng[keep_mask]

    win = np.hanning(NUM_ADC_SAMPLES).astype(np.float32)
    sos_radar = make_bandpass(BP_LO, BP_HI, FPS_HZ)
    sos_belt  = make_bandpass(BELT_BP_LO, BELT_BP_HI, BELT_NOMINAL_HZ)

    frame_count = 0
    bg_accum = np.zeros((n_bins,), dtype=np.complex128)

    # Breathing-detection phase: accumulate per-bin phase for DETECT_FRAMES
    detect_phase_buf = []          # list of (n_bins,) phase arrays
    detect_prev_phase = np.zeros(n_bins)  # for per-bin unwrap during detection
    bg_saved = None                # background saved after BG_FRAMES

    radar_phase_hist  = collections.deque(maxlen=MAX_HISTORY)
    radar_disp_hist   = collections.deque(maxlen=MAX_HISTORY)
    radar_time_hist   = collections.deque(maxlen=MAX_HISTORY)
    range_profile_db  = np.full(keep_mask.sum(), -60.0)

    target_bin = None
    radar_bpm  = 0.0
    prev_phase = 0.0

    belt_bpm   = 0.0

    # ── 5. Figure layout ───────────────────────────────────────────
    nrows = 3 if radar_enabled else 2
    height_ratios = [1, 1, 1] if radar_enabled else [1, 1]

    fig = plt.figure(figsize=(12, 8 if radar_enabled else 5.5))
    fig.patch.set_facecolor("#1a1a2e")

    if radar_enabled:
        gs = GridSpec(3, 3, width_ratios=[3, 1, 1],
                      height_ratios=[1, 1, 1],
                      hspace=0.40, wspace=0.30)
    else:
        gs = GridSpec(2, 3, width_ratios=[3, 1, 1],
                      height_ratios=[1, 1],
                      hspace=0.40, wspace=0.30)

    # ── styling helper ──
    def style_ax(ax, title="", ylabel=""):
        ax.set_facecolor("#16213e")
        if title:
            ax.set_title(title, color="white", fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.15, color="white")
        for sp in ax.spines.values():
            sp.set_color("#333")

    def style_bpm_ax(ax, label, color):
        ax.set_facecolor("#16213e")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333")
        txt_val = ax.text(0.5, 0.55, "--", transform=ax.transAxes,
                          fontsize=42, fontweight="bold", color=color,
                          ha="center", va="center", family="monospace")
        ax.text(0.5, 0.10, label, transform=ax.transAxes,
                fontsize=12, color="#aaaaaa", ha="center", va="center")
        return txt_val

    # Row indices
    row = 0

    if radar_enabled:
        # ── range profile (top-left) ──
        ax_rng = fig.add_subplot(gs[0, 0])
        style_ax(ax_rng, "range profile", "|H−bg| (dB)")
        line_rng, = ax_rng.plot(rng_plot, range_profile_db, color="#0ff", lw=1.4)
        vline_target = ax_rng.axvline(0.5, color="#ff6361", ls="--", lw=0.9, alpha=0.7)
        ax_rng.set_xlim(0, XMAX_M)
        ax_rng.set_ylim(-60, 40)
        ax_rng.set_xlabel("range (m)", color="white")

        txt_status = ax_rng.text(0.98, 0.92, "calibrating…",
                                 transform=ax_rng.transAxes, fontsize=9,
                                 color="#ff9f43", ha="right", va="top")

        # ── radar BPM (top-center) ──
        ax_rbpm = fig.add_subplot(gs[0, 1])
        txt_rbpm = style_bpm_ax(ax_rbpm, "radar BPM", "#00ff99")

        # ── belt BPM (top-right) ──
        ax_bbpm = fig.add_subplot(gs[0, 2])
        txt_bbpm = style_bpm_ax(ax_bbpm, "belt BPM", "#ff79c6")

        # ── radar displacement (middle row) ──
        ax_disp = fig.add_subplot(gs[1, :])
        style_ax(ax_disp, "radar chest displacement  (0.1–0.6 Hz)",
                 "displacement (mm)")
        line_disp, = ax_disp.plot([], [], color="#feca57", lw=1.2)
        cursor_disp = ax_disp.axvline(0, color="#ff6361", lw=1)
        ax_disp.set_xlim(0, HISTORY_S)
        ax_disp.set_ylim(-2, 2)
        ax_disp.set_xlabel("time (s)", color="white")

        # ── belt force (bottom row) ──
        ax_belt = fig.add_subplot(gs[2, :])
    else:
        # Belt-only mode: just BPM + waveform
        ax_bbpm = fig.add_subplot(gs[0, :])
        txt_bbpm = style_bpm_ax(ax_bbpm, "belt BPM", "#ff79c6")
        ax_belt = fig.add_subplot(gs[1, :])
        # placeholders so update() doesn't crash
        txt_rbpm = None
        txt_status = None

    style_ax(ax_belt, "belt force waveform", "force (N)")
    line_belt, = ax_belt.plot([], [], color="#ff79c6", lw=1.2)
    cursor_belt = ax_belt.axvline(0, color="#ff6361", lw=1)
    ax_belt.set_xlim(0, BELT_HISTORY_S)
    ax_belt.set_ylim(0, 15)
    ax_belt.set_xlabel("time (s)", color="white")

    # ── 6. Animation callback ──────────────────────────────────────
    def update(_frame_unused):
        nonlocal frame_count, bg_accum, target_bin, radar_bpm, prev_phase
        nonlocal range_profile_db, belt_bpm, bg_saved, detect_prev_phase

        # ════ RADAR PROCESSING ════
        if radar_enabled:
            frames_this_tick = []
            with buf_lock:
                while len(raw_buf) >= RAW_BYTES_PER_FRAME:
                    chunk = bytes(raw_buf[:RAW_BYTES_PER_FRAME])
                    del raw_buf[:RAW_BYTES_PER_FRAME]
                    frames_this_tick.append(chunk)

            for raw_chunk in frames_this_tick:
                frame = parse_one_frame(raw_chunk)
                wall_now = time.monotonic()

                dc = frame.mean(axis=-1, keepdims=True)
                windowed = (frame - dc) * win
                rfft = np.fft.fft(windowed, axis=-1)[..., :n_bins]
                rfft_avg = rfft.mean(axis=0).squeeze()

                frame_count += 1
                t_rel = wall_now - t_epoch

                # ── Phase 1: background accumulation (frames 1..BG_FRAMES) ──
                if frame_count <= BG_FRAMES:
                    bg_accum += rfft_avg
                    pct = frame_count / BG_FRAMES * 100
                    if txt_status:
                        txt_status.set_text(f"calibrating… {pct:.0f}%")
                    continue

                # Save background once
                if bg_saved is None:
                    bg_saved = bg_accum / BG_FRAMES
                    if txt_status:
                        txt_status.set_text("detecting breathing…")
                    print("[RADAR] bg done — now detecting breathing target…")

                # ── Phase 2: breathing detection (next DETECT_FRAMES) ──
                detect_idx = frame_count - BG_FRAMES  # 1-based within detection
                if detect_idx <= DETECT_FRAMES:
                    # Unwrap phase at every candidate bin
                    raw_ph = np.angle(rfft_avg[:n_bins])
                    diff_arr = raw_ph - detect_prev_phase
                    diff_arr[diff_arr > np.pi] -= 2 * np.pi
                    diff_arr[diff_arr < -np.pi] += 2 * np.pi
                    detect_prev_phase = detect_prev_phase + diff_arr
                    detect_phase_buf.append(detect_prev_phase.copy())

                    pct = detect_idx / DETECT_FRAMES * 100
                    if txt_status:
                        txt_status.set_text(f"detecting breathing… {pct:.0f}%")

                    # Update range profile for visual feedback
                    mag = np.abs(rfft_avg - bg_saved)
                    mag_db = 20 * np.log10(mag + 1e-6)
                    range_profile_db = mag_db[keep_mask]
                    continue

                # ── Phase 2 → 3 transition: pick the breathing bin ──
                if target_bin is None:
                    phase_mat = np.array(detect_phase_buf)  # (DETECT_FRAMES, n_bins)
                    mag_profile = np.abs(bg_saved)
                    target_bin, scores = find_breathing_bin(
                        phase_mat, FPS_HZ, rng,
                        min_range=0.2, max_range=1.5,
                        mag_profile=mag_profile,
                    )
                    vline_target.set_xdata([rng[target_bin], rng[target_bin]])
                    if txt_status:
                        txt_status.set_text(f"tracking @ {rng[target_bin]:.2f} m")

                    # Log the top 3 candidates for debugging
                    lo_b = max(1, int(np.searchsorted(rng, 0.2)))
                    hi_b = int(np.searchsorted(rng, 1.5))
                    ranked = np.argsort(scores[lo_b:hi_b])[::-1] + lo_b
                    print(f"[RADAR] breathing-bin scores (top 3):")
                    for r in ranked[:3]:
                        print(f"  bin {r:3d}  ({rng[r]:.3f} m)  score={scores[r]:.1f}")
                    print(f"[RADAR] → selected bin {target_bin} "
                          f"({rng[target_bin]:.3f} m)")

                    # Initialise phase tracking from the last detection frame
                    prev_phase = detect_prev_phase[target_bin]

                # ── Phase 3: normal tracking ──
                bg = bg_saved
                mag = np.abs(rfft_avg - bg)
                mag_db = 20 * np.log10(mag + 1e-6)
                range_profile_db = mag_db[keep_mask]

                z = rfft_avg[target_bin]
                ang = np.angle(z)
                diff = ang - prev_phase
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                prev_phase = prev_phase + diff
                unwrapped = prev_phase

                radar_phase_hist.append(unwrapped)
                radar_time_hist.append(t_rel)

                if len(radar_phase_hist) > 10:
                    ph = np.array(radar_phase_hist)
                    ts = np.array(radar_time_hist)
                    a, b = np.polyfit(ts, ph, 1)
                    ph_dt = ph - (a * ts + b)
                    disp_mm = -LAMBDA_M * ph_dt / (4 * np.pi) * 1000.0

                    if len(disp_mm) > 27:
                        disp_filt = sosfiltfilt(sos_radar, disp_mm)
                    else:
                        disp_filt = disp_mm

                    radar_disp_hist.clear()
                    radar_disp_hist.extend(disp_filt)

                    win_n = min(len(disp_filt), int(BPM_WINDOW_S * FPS_HZ))
                    radar_bpm, _ = estimate_bpm(disp_filt[-win_n:], FPS_HZ)

            # ── update radar plots ──
            if target_bin is not None:
                line_rng.set_ydata(range_profile_db)
                lo_db = float(np.min(range_profile_db)) - 3
                hi_db = float(np.max(range_profile_db)) + 3
                ax_rng.set_ylim(lo_db, hi_db)

                if radar_bpm > 0 and txt_rbpm:
                    txt_rbpm.set_text(f"{radar_bpm:.0f}")
                    if 10 <= radar_bpm <= 24:
                        txt_rbpm.set_color("#00ff99")
                    elif radar_bpm < 10:
                        txt_rbpm.set_color("#feca57")
                    else:
                        txt_rbpm.set_color("#ff6361")

                if len(radar_disp_hist) > 1:
                    ts_arr = np.array(radar_time_hist)
                    ds_arr = np.array(radar_disp_hist)
                    line_disp.set_data(ts_arr, ds_arr)
                    ax_disp.set_xlim(max(0, ts_arr[-1] - HISTORY_S),
                                     ts_arr[-1] + 0.5)
                    margin = max(float(np.abs(ds_arr).max()) * 1.3, 0.5)
                    ax_disp.set_ylim(-margin, margin)
                    cursor_disp.set_xdata([ts_arr[-1], ts_arr[-1]])

        # ════ BELT PROCESSING ════
        with belt_lock:
            if len(belt_data) < 2:
                return
            belt_snap = list(belt_data)  # snapshot

        belt_t = np.array([s[0] - t_epoch for s in belt_snap])
        belt_f = np.array([s[1] for s in belt_snap])

        # ── belt BPM (spectral, same approach as radar) ──
        if len(belt_f) > 30:
            # estimate actual sample rate from timestamps
            dt_median = float(np.median(np.diff(belt_t)))
            belt_fs = 1.0 / dt_median if dt_median > 0 else BELT_NOMINAL_HZ

            # bandpass the force signal
            if len(belt_f) > 27:
                try:
                    sos_b = make_bandpass(BELT_BP_LO, BELT_BP_HI, belt_fs)
                    belt_filt = sosfiltfilt(sos_b, belt_f - belt_f.mean())
                except Exception:
                    belt_filt = belt_f - belt_f.mean()
            else:
                belt_filt = belt_f - belt_f.mean()

            win_n = min(len(belt_filt), int(BELT_BPM_WINDOW_S * belt_fs))
            belt_bpm, _ = estimate_bpm(belt_filt[-win_n:], belt_fs)

        if belt_bpm > 0 and txt_bbpm:
            txt_bbpm.set_text(f"{belt_bpm:.0f}")
            if 10 <= belt_bpm <= 24:
                txt_bbpm.set_color("#ff79c6")
            elif belt_bpm < 10:
                txt_bbpm.set_color("#feca57")
            else:
                txt_bbpm.set_color("#ff6361")

        # ── belt waveform plot ──
        line_belt.set_data(belt_t, belt_f)
        ax_belt.set_xlim(max(0, belt_t[-1] - BELT_HISTORY_S),
                         belt_t[-1] + 0.5)
        f_min = float(belt_f.min()) - 1
        f_max = float(belt_f.max()) + 1
        ax_belt.set_ylim(f_min, f_max)
        cursor_belt.set_xdata([belt_t[-1], belt_t[-1]])

    # ── 7. Launch ──────────────────────────────────────────────────
    ani = FuncAnimation(fig, update, interval=40, blit=False,
                        cache_frame_data=False)
    print("\n[MAIN] dashboard running — close the window or Ctrl+C to stop\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    # ── 8. Cleanup ─────────────────────────────────────────────────
    belt_reader.stop()
    if radar_src:
        radar_src.stop()
    belt_reader.join(timeout=3)
    if radar_src:
        radar_src.join(timeout=2)
    godirect_ctx.quit()
    print("\n[DONE]")


if __name__ == "__main__":
    main()