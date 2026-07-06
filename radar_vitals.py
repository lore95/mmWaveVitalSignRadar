#!/usr/bin/env python3
"""
Multi-target real-time breathing monitor — 2TX/4RX TDM-MIMO
────────────────────────────────────────────────────────────
Features:
  • 2D range-angle heatmap from 8-element virtual ULA
  • Multi-target Kalman tracking with per-person BPM
  • Optional Vernier respiration belt for ground truth
  • 10-second CFAR background calibration

Layout (with belt):                    Layout (no belt):
┌──────────────┬──────────┐           ┌──────────────┬──────────┐
│ range-angle  │ BPM per  │           │ range-angle  │ BPM per  │
│ heatmap (2D) │ track    │           │ heatmap (2D) │ track    │
├──────────────┴──────────┤           ├──────────────┴──────────┤
│ displacement waveforms  │           │ displacement waveforms  │
├─────────────────────────┤           └─────────────────────────┘
│ belt force waveform     │
└─────────────────────────┘

Usage:
  python multi_breathing_monitor.py --demo
  python multi_breathing_monitor.py
  python multi_breathing_monitor.py --no-radar   # belt only
"""

import argparse
import threading
import socket
import time
import sys
import collections

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, sosfiltfilt

from kalman_tracker import KalmanPeakTracker, interpolate_complex_at
from multi_track_manager import MultiTrackManager, TRACK_COLORS

# ═══════════════════════════════════════════════════════════════════════
# Radar constants — 2TX / 4RX TDM-MIMO
# ═══════════════════════════════════════════════════════════════════════
NUM_RX            = 4
NUM_TX            = 3           # TX0 + TX1 + TX2 all enabled
NUM_CHIRPS        = 64          # CHIRP_LOOPS = 64 per TX per frame
NUM_ADC_SAMPLES   = 256         # was 64
FRAME_PERIOD_S    = 20e-3       # 20 ms periodicity = 50 Hz
FPS_HZ            = 1.0 / FRAME_PERIOD_S

C                 = 2.998e8
SLOPE_HZ_PER_S   = 78.020e12   # 78.020 MHz/µs
SAMPLE_RATE_HZ    = 10e6        # 10000 ksps (was 3.6e6)
F0_HZ             = 77.0000000238419e9
LAMBDA_M          = C / F0_HZ

TOTAL_CHIRPS      = NUM_CHIRPS * NUM_TX     # 64 × 3 = 192 chirps per frame
COMPLEX_PER_FRAME = TOTAL_CHIRPS * NUM_RX * NUM_ADC_SAMPLES   # 192 × 4 × 256 = 196,608
# Complex1x format = 4 bytes per complex sample (I + Q as int16)
RAW_BYTES_PER_FRAME = COMPLEX_PER_FRAME * 4    # 786,432 bytes (0.75 MB)

NUM_VIRTUAL       = NUM_TX * NUM_RX  # 12 virtual elements
AOA_FFT_SIZE      = 64

BG_FRAMES         = 500         # 10 s at 50 Hz
HISTORY_S         = 45
MAX_HISTORY       = int(HISTORY_S * FPS_HZ)
BP_LO, BP_HI     = 0.04, 0.6   # 2.4 – 36 BPM (covers down to ~3 BPM)
BPM_WINDOW_S      = 40          # 40s window — 2 full cycles at 3 BPM
BPM_REFRESH_S     = 5.0        # recompute BPM every 5 seconds (not every frame)
XMAX_M            = 4.0        # display max range (radar max is 9.5m but breathing range is closer)
ANGLE_MAX_DEG     = 60         # angle axis limits for heatmap

HOST_IP           = "192.168.33.30"
DATA_PORT         = 4098

MAX_TRACKS        = 4
SEARCH_MIN_M      = 0.7        # exclude near-field TX-RX coupling region (0-0.7m)
SEARCH_MAX_M      = 4.0
CONFIRM_FRAMES    = 3
DELETE_FRAMES     = 150         # 3 seconds at 50 Hz — coast through brief misses
MIN_PEAK_SEP_BINS = 6     # ~45 cm at 7.5 cm/bin (person body diameter)
MIN_TRACK_SEP_BINS = 4    # ~30 cm
SNR_THRESHOLD_DB  = 10.0
CFAR_K            = 4.0

# Adaptive background update (subtracts slow drift after calibration)
# alpha = how much new data influences the running background each frame.
# At 50 Hz, alpha=0.005 → ~10-second time constant.
# Breathing (~0.04-0.6 Hz) passes through unaffected.
# Slow clutter drift (object placement, temperature) gets removed.
# Adaptive background — DISABLED because the environment is guaranteed
# static after the 10-second calibration. New objects appearing after
# calibration are the targets we want to track (people entering).
# Keeping the background frozen means new arrivals stay visible indefinitely.
ADAPTIVE_BG_ALPHA = 0.005
ADAPTIVE_BG_ENABLED = False

KALMAN_Q_POS      = 0.1
KALMAN_Q_VEL      = 0.01
KALMAN_R_MEAS     = 1.0
KALMAN_GATE_SIGMA = 5.0

BELT_HISTORY_S    = 30
BELT_NOMINAL_HZ   = 10
MAX_BELT_HISTORY  = int(BELT_HISTORY_S * BELT_NOMINAL_HZ)
BELT_BPM_WINDOW_S = 40
BELT_BP_LO        = 0.04
BELT_BP_HI        = 0.6


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def range_axis(n: int) -> np.ndarray:
    return np.arange(n // 2) * (SAMPLE_RATE_HZ / n) * C / (2 * SLOPE_HZ_PER_S)

def make_bandpass(lo, hi, fs, order=4):
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")

def estimate_bpm(signal_seg, fs):
    if len(signal_seg) < 64:
        return 0.0, 0.0
    sig = signal_seg - signal_seg.mean()
    f = np.fft.rfftfreq(len(sig), d=1 / fs)
    mag = np.abs(np.fft.rfft(sig))
    band = (f >= BP_LO) & (f <= BP_HI)
    if not band.any():
        return 0.0, 0.0
    return float(f[band][np.argmax(mag[band])]) * 60.0, 0.0

def angle_axis(n_fft: int = AOA_FFT_SIZE) -> np.ndarray:
    """Map FFT bins to angle in degrees for λ/2-spaced ULA."""
    u = np.linspace(-1, 1, n_fft, endpoint=False)
    u = np.clip(u, -1, 1)
    return np.degrees(np.arcsin(u))


# ═══════════════════════════════════════════════════════════════════════
# Frame parsing — 2TX / 4RX
# ═══════════════════════════════════════════════════════════════════════

def parse_one_frame(raw_bytes: bytes) -> np.ndarray:
    """Parse Complex1x format frame: 192 chirps × 4 RX × 256 samples × 4 bytes.

    Frame layout: 64 loops, each loop contains 3 chirps (TX1, TX0, TX2 order).
    Total 192 chirps per frame interleaved by TX.

    The captured signal is the complex conjugate of the intended
    representation — targets appear at negative frequencies. We swap I
    and Q (equivalent to conjugating + shifting) so that closer targets
    map to lower range bins.
    """
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    # IQ swap: put Q in real part, I in imaginary part
    iq = raw[1::2].astype(np.float32) + 1j * raw[0::2].astype(np.float32)
    return iq.reshape(TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES)

def separate_tx(frame: np.ndarray):
    """Separate the 192 interleaved chirps by TX.

    Per Lua chirp config, each loop fires chirps in this order:
      Chirp 0 → TX1 (elevated)
      Chirp 1 → TX0
      Chirp 2 → TX2

    With 64 loops, frame layout is:
      [TX1_0, TX0_0, TX2_0, TX1_1, TX0_1, TX2_1, ..., TX1_63, TX0_63, TX2_63]

    Returns:
        tx0: (64, 4, 256) — all TX0 chirps
        tx1: (64, 4, 256) — all TX1 chirps (elevation antenna, used for SNR only)
        tx2: (64, 4, 256) — all TX2 chirps
    """
    tx1 = frame[0::3]   # chirps 0, 3, 6, ... = TX1
    tx0 = frame[1::3]   # chirps 1, 4, 7, ... = TX0
    tx2 = frame[2::3]   # chirps 2, 5, 8, ... = TX2
    return tx0, tx1, tx2

def compute_range_fft(chirps: np.ndarray, win: np.ndarray) -> np.ndarray:
    dc = chirps.mean(axis=-1, keepdims=True)
    windowed = (chirps - dc) * win
    rfft = np.fft.fft(windowed, axis=-1)[..., :NUM_ADC_SAMPLES // 2]
    return rfft.mean(axis=0)  # (n_rx, n_bins) — coherently averaged over 64 chirps

def compute_range_angle_map(virtual: np.ndarray, n_fft: int = AOA_FFT_SIZE) -> np.ndarray:
    """Compute range-angle heatmap from virtual array data.

    Args:
        virtual: (8, n_bins) complex — bg-subtracted virtual array (TX0+TX2 for azimuth)
        n_fft: angle FFT size

    Returns:
        (n_fft, n_bins) magnitude in dB
    """
    angle_fft = np.fft.fftshift(np.fft.fft(virtual, n=n_fft, axis=0), axes=0)
    mag = np.abs(angle_fft)
    return 20 * np.log10(mag + 1e-6)

def estimate_aoa(rfft_tx0: np.ndarray, rfft_tx2: np.ndarray,
                 bin_idx: int, n_fft: int = AOA_FFT_SIZE) -> float:
    """Azimuth AoA from TX0 + TX2 virtual array (8 elements at λ/2 spacing).

    On the XWR1843, TX0 and TX2 are 2λ apart horizontally,
    giving a uniform 8-element ULA when combined with the 4 RX.
    """
    idx = min(bin_idx, rfft_tx0.shape[1] - 1)
    virtual = np.zeros(8, dtype=np.complex64)
    virtual[0:4] = rfft_tx0[:, idx]
    virtual[4:8] = rfft_tx2[:, idx]
    spectrum = np.fft.fftshift(np.fft.fft(virtual, n=n_fft))
    angles = angle_axis(n_fft)
    peak_idx = int(np.argmax(np.abs(spectrum)))
    return float(angles[peak_idx])


# ═══════════════════════════════════════════════════════════════════════
# Belt (optional)
# ═══════════════════════════════════════════════════════════════════════

def ask_belt() -> bool:
    """Ask user if they want to connect the respiration belt."""
    print("\n  Respiration belt (optional):")
    try:
        ans = input("  Connect GDX belt? [y/N]: ").strip().lower()
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
    def __init__(self, device, data_deque, lock, period_ms=100):
        super().__init__(daemon=True)
        self.device, self.data, self.lock = device, data_deque, lock
        self.period_ms = period_ms
        self.running = True
        self.t0 = None

    def run(self):
        dev = self.device
        if not dev.open():
            print("[BELT] ERROR: could not open device")
            self.running = False
            return
        dev.enable_default_sensors()
        enabled = dev.get_enabled_sensors()
        fs = None
        for s in enabled:
            if "force" in s.sensor_description.lower():
                fs = s
                break
        if fs is None and enabled:
            fs = enabled[0]
        print(f"[BELT] streaming {fs.sensor_description} @ {1000/self.period_ms:.0f} Hz")
        dev.start(period=self.period_ms)
        self.t0 = time.monotonic()
        while self.running:
            try:
                if dev.read():
                    v = fs.value
                    if v is not None and v == v:
                        with self.lock:
                            self.data.append((time.monotonic(), float(v)))
            except Exception as e:
                if self.running:
                    print(f"[BELT] error: {e}")
                break
        try:
            dev.stop(); dev.close()
        except Exception:
            pass

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════
# UDP capture thread
# ═══════════════════════════════════════════════════════════════════════

class UDPCapture(threading.Thread):
    def __init__(self, host, port, buf, lock):
        super().__init__(daemon=True)
        self.host, self.port, self.buf, self.lock = host, port, buf, lock
        self.running = True

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
# Demo source — 2TX/4RX, 2 targets at different angles
# ═══════════════════════════════════════════════════════════════════════

class DemoSource(threading.Thread):
    def __init__(self, buf, lock, bpm1=14.0, bpm2=20.0):
        super().__init__(daemon=True)
        self.buf, self.lock = buf, lock
        self.running = True
        self.bpm1, self.bpm2 = bpm1, bpm2
        self.frame_idx = 0

    def run(self):
        rng = range_axis(NUM_ADC_SAMPLES)
        bin1 = int(np.searchsorted(rng, 0.6))
        bin2 = int(np.searchsorted(rng, 1.5))
        f1, f2 = self.bpm1 / 60.0, self.bpm2 / 60.0
        a1, a2 = np.radians(-10), np.radians(15)
        vp = np.arange(NUM_VIRTUAL)
        n_adc = np.arange(NUM_ADC_SAMPLES)

        print(f"[DEMO] 3TX/4RX MIMO — 2 targets (Complex1x, 64 loops/TX):")
        print(f"  #1: {rng[bin1]:.2f}m, {self.bpm1:.0f} BPM, θ=-10°")
        print(f"  #2: {rng[bin2]:.2f}m, {self.bpm2:.0f} BPM, θ=+15°")

        # Pre-compute virtual element azimuth phase shifts (use TX0+TX2 for azimuth)
        # TX1 is elevation antenna — give it small random offset to not contribute to azimuth
        # vp indexing: vi = tx*4 + rx, where tx_order in frame is [TX1, TX0, TX2]
        # For azimuth: TX0 contributes elements 0-3 at positions 0..3
        #              TX2 contributes elements 4-7 at positions 4..7
        # TX1 (elevation) is not used for azimuth — gets a different phase

        while self.running:
            t = self.frame_idx * FRAME_PERIOD_S
            drift1 = 2.0 * np.sin(2 * np.pi * t / 30.0)
            tbin1 = bin1 + drift1
            ph1 = 0.005 * np.sin(2 * np.pi * f1 * t)
            ph2 = 0.004 * np.sin(2 * np.pi * f2 * t)

            frame_iq = np.zeros((TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES),
                                dtype=np.complex64)

            # Chirp order in frame: [TX1, TX0, TX2, TX1, TX0, TX2, ...] × 64 loops
            tx_order = [1, 0, 2]  # which TX fires on each chirp index within a loop

            for loop in range(NUM_CHIRPS):  # 64 loops
                for chirp_in_loop in range(3):
                    ci = loop * 3 + chirp_in_loop
                    tx_id = tx_order[chirp_in_loop]
                    for rx in range(NUM_RX):
                        # Virtual element position for azimuth (TX0+TX2 only)
                        if tx_id == 0:
                            virt_pos = rx               # 0..3
                        elif tx_id == 2:
                            virt_pos = 4 + rx           # 4..7
                        else:
                            virt_pos = rx               # TX1 same as TX0 baseline
                        s1 = 2*np.pi * virt_pos * 0.5 * np.sin(a1)
                        s2 = 2*np.pi * virt_pos * 0.5 * np.sin(a2)
                        t1 = 800*np.exp(1j*(2*np.pi*tbin1/NUM_ADC_SAMPLES*n_adc + ph1 + s1))
                        t2 = 600*np.exp(1j*(2*np.pi*bin2/NUM_ADC_SAMPLES*n_adc + ph2 + s2))
                        noise = (np.random.randn(NUM_ADC_SAMPLES) +
                                 1j*np.random.randn(NUM_ADC_SAMPLES)) * 5
                        frame_iq[ci, rx, :] = t1 + t2 + noise

            # Pack as Complex1x: alternating I and Q int16s (no zero padding)
            flat = frame_iq.reshape(-1)
            raw = np.empty(len(flat) * 2, dtype=np.int16)
            # Need to match the IQ swap done in parser: parser does
            #   iq = raw[1::2] + 1j * raw[0::2]
            # So we need to store imag at [0::2] and real at [1::2]
            raw[0::2] = np.round(flat.imag).astype(np.int16)
            raw[1::2] = np.round(flat.real).astype(np.int16)
            with self.lock:
                self.buf.extend(raw.tobytes())
            self.frame_idx += 1
            time.sleep(FRAME_PERIOD_S)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Multi-target breathing monitor (3TX/4RX MIMO)")
    ap.add_argument("--host", default=HOST_IP)
    ap.add_argument("--port", type=int, default=DATA_PORT)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--bpm1", type=float, default=14.0)
    ap.add_argument("--bpm2", type=float, default=20.0)
    ap.add_argument("--no-radar", action="store_true", help="belt only, no radar")
    ap.add_argument("--belt-period", type=int, default=100)
    args = ap.parse_args()

    # ── 1. Optional belt ───────────────────────────────────────────
    use_belt = False
    godirect_ctx = None
    belt_reader = None
    belt_data = collections.deque(maxlen=MAX_BELT_HISTORY)
    belt_lock = threading.Lock()

    if ask_belt():
        godirect_ctx, belt_device = select_belt_device()
        if godirect_ctx and belt_device:
            belt_reader = BeltReader(belt_device, belt_data, belt_lock,
                                     period_ms=args.belt_period)
            use_belt = True
        else:
            print("  Proceeding without belt.")

    # ── 2. Radar source ───────────────────────────────────────────
    radar_enabled = not args.no_radar
    raw_buf = bytearray()
    buf_lock = threading.Lock()
    radar_src = None
    if radar_enabled:
        if args.demo:
            radar_src = DemoSource(raw_buf, buf_lock,
                                   bpm1=args.bpm1, bpm2=args.bpm2)
        else:
            radar_src = UDPCapture(args.host, args.port, raw_buf, buf_lock)

    if not radar_enabled and not use_belt:
        print("ERROR: no radar and no belt — nothing to do.")
        sys.exit(1)

    # ── 3. Start ──────────────────────────────────────────────────
    if radar_enabled:
        print("\n" + "═" * 55)
        print("  ENVIRONMENT CALIBRATION  (2TX / 4RX MIMO)")
        print("  Ensure the monitored area is EMPTY of people.")
        print("  10-second static background recording.")
        print("═" * 55)
        input("  Press Enter when the area is clear… ")
        print()

    if use_belt:
        belt_reader.start()
    if radar_src:
        radar_src.start()

    # Epoch: belt t0 or now
    if use_belt:
        print("[MAIN] waiting for belt …")
        deadline = time.monotonic() + 15
        while belt_reader.t0 is None and time.monotonic() < deadline:
            time.sleep(0.1)
        t_epoch = belt_reader.t0 if belt_reader.t0 else time.monotonic()
    else:
        t_epoch = time.monotonic()

    # ── 4. Radar state ────────────────────────────────────────────
    rng = range_axis(NUM_ADC_SAMPLES)
    n_bins = NUM_ADC_SAMPLES // 2
    keep_mask = rng <= XMAX_M
    rng_plot = rng[keep_mask]
    n_keep = keep_mask.sum()
    angles = angle_axis(AOA_FFT_SIZE)
    angle_mask = np.abs(angles) <= ANGLE_MAX_DEG
    angles_plot = angles[angle_mask]

    win_hann = np.hanning(NUM_ADC_SAMPLES).astype(np.float32)

    frame_count = 0
    bg_frames_buf = []       # list of (12, n_bins) virtual arrays for calibration
    bg_virtual_saved = None  # (12, n_bins) complex background
    bg_noise_std = None      # (n_bins,) per-bin noise std

    last_rfft_tx0 = None
    last_rfft_tx2 = None

    search_min_bin = max(1, int(np.searchsorted(rng, SEARCH_MIN_M)))
    search_max_bin = min(n_bins - 1, int(np.searchsorted(rng, SEARCH_MAX_M)))

    track_mgr = None
    track_angles = {}

    range_profile_db = np.full(n_keep, -60.0)
    # Range-angle map: (n_angles_visible, n_range_visible)
    ra_map_db = np.full((angle_mask.sum(), n_keep), -60.0)
    belt_bpm = 0.0
    last_belt_bpm_time = 0.0     # wall time of last belt BPM recomputation

    # ── 5. Figure layout ──────────────────────────────────────────
    n_rows = 3 if use_belt else 2
    h_ratios = [1.8, 1, 0.8] if use_belt else [1.8, 1]
    fig_h = 11 if use_belt else 8.5

    fig = plt.figure(figsize=(14, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(n_rows, 2, width_ratios=[3, 1],
                  height_ratios=h_ratios,
                  hspace=0.35, wspace=0.25)

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

    # Radar-style colourmap: dark → green → yellow → orange → red
    cmap_ra = LinearSegmentedColormap.from_list("radar_fan", [
        "#0a0a1a", "#0a2a0a", "#1a4a1a", "#2d6b2d",
        "#4a8a2a", "#7ab030", "#b8d430",
        "#e8e020", "#f0a010", "#e05010", "#cc1010"
    ])

    # Pre-compute polar meshgrid for fan-shaped display
    r_edges = np.linspace(0, XMAX_M, n_keep + 1)
    a_edges_deg = np.linspace(-ANGLE_MAX_DEG, ANGLE_MAX_DEG, angle_mask.sum() + 1)
    a_edges_rad = np.radians(a_edges_deg)
    R_mesh, A_mesh = np.meshgrid(r_edges, a_edges_rad)
    X_mesh = R_mesh * np.sin(A_mesh)   # lateral position
    Y_mesh = R_mesh * np.cos(A_mesh)   # forward (boresight) position

    if radar_enabled:
        # ── Fan-shaped radar display (top-left) ──
        ax_ra = fig.add_subplot(gs[0, 0])
        ax_ra.set_facecolor("#0a0a1a")
        ax_ra.set_title("radar monitor  (3TX/4RX MIMO)", color="white", fontsize=11)
        ax_ra.set_xlabel("lateral (m)", color="white")
        ax_ra.set_ylabel("range (m)", color="white")
        ax_ra.tick_params(colors="white")
        for sp in ax_ra.spines.values():
            sp.set_visible(False)

        # Fan-shaped pcolormesh
        pcm_ra = ax_ra.pcolormesh(X_mesh, Y_mesh, ra_map_db,
                                   cmap=cmap_ra, vmin=-40, vmax=20,
                                   shading="flat")

        # Range rings
        ring_theta = np.linspace(-np.radians(ANGLE_MAX_DEG),
                                  np.radians(ANGLE_MAX_DEG), 200)
        for r_ring in range(1, int(XMAX_M) + 1):
            ax_ra.plot(r_ring * np.sin(ring_theta), r_ring * np.cos(ring_theta),
                       color="white", lw=0.4, alpha=0.35)
            ax_ra.text(0.08, r_ring - 0.08, f"{r_ring}m",
                       color="white", fontsize=7, alpha=0.5)

        # Angle spokes every 20°
        for a_spoke in range(-40, 41, 20):
            a_rad = np.radians(a_spoke)
            ax_ra.plot([0, XMAX_M * np.sin(a_rad)],
                       [0, XMAX_M * np.cos(a_rad)],
                       color="white", lw=0.3, alpha=0.25)
            if a_spoke != 0:
                ax_ra.text(XMAX_M * 0.98 * np.sin(a_rad),
                           XMAX_M * 0.98 * np.cos(a_rad),
                           f"{a_spoke}°", color="white", fontsize=7,
                           alpha=0.4, ha="center", va="bottom")

        # Radar origin marker
        ax_ra.plot(0, 0, '^', color="#00ff99", markersize=6, zorder=6)

        # Set aspect and limits for fan shape
        ax_ra.set_aspect("equal")
        x_fan_max = XMAX_M * np.sin(np.radians(ANGLE_MAX_DEG)) * 1.05
        ax_ra.set_xlim(-x_fan_max, x_fan_max)
        ax_ra.set_ylim(-0.15, XMAX_M * 1.03)

        cbar = fig.colorbar(pcm_ra, ax=ax_ra, pad=0.02, fraction=0.04)
        cbar.set_label("dB", color="white", fontsize=9)
        cbar.ax.tick_params(colors="white")

        # Tracked target scatter overlay (in cartesian coords) — used only for
        # pending (grey) tracks. Validated tracks get colored patches instead.
        scatter_tracks = ax_ra.scatter([], [], s=100, edgecolors="white",
                                        linewidths=2, zorder=5, c=[])

        # Filled arc patches drawn over the heatmap for VALIDATED tracks.
        # We maintain a list and clear/redraw each frame.
        track_patches = []

        txt_status = ax_ra.text(0.98, 0.95, "calibrating…",
                                transform=ax_ra.transAxes, fontsize=9,
                                color="#ff9f43", ha="right", va="top")

        # ── BPM panel (top-right) ──
        ax_bpm = fig.add_subplot(gs[0, 1])
        ax_bpm.set_facecolor("#16213e")
        ax_bpm.set_xticks([]); ax_bpm.set_yticks([])
        for sp in ax_bpm.spines.values():
            sp.set_color("#333")

        n_slots = MAX_TRACKS + (1 if use_belt else 0)
        bpm_texts = []
        for i in range(n_slots):
            y_pos = 1.0 - (i + 0.5) / n_slots
            if i < MAX_TRACKS:
                color = TRACK_COLORS[i]
                label = f"track {i+1}"
            else:
                color = "#ff79c6"
                label = "belt"
            tv = ax_bpm.text(0.50, y_pos + 0.02, "--",
                             transform=ax_bpm.transAxes,
                             fontsize=28, fontweight="bold", color=color,
                             ha="center", va="center", family="monospace")
            ax_bpm.text(0.50, y_pos - 0.06, label,
                        transform=ax_bpm.transAxes,
                        fontsize=9, color="#777777", ha="center", va="center")
            bpm_texts.append(tv)

        # ── Displacement waveforms (row 1) ──
        ax_disp = fig.add_subplot(gs[1, :])
        style_ax(ax_disp, "chest displacement per track  (0.1–0.6 Hz)",
                 "displacement (mm)")
        ax_disp.set_xlim(0, HISTORY_S)
        ax_disp.set_ylim(-2, 2)
        ax_disp.set_xlabel("time (s)", color="white")

        disp_lines = []
        for i in range(MAX_TRACKS):
            ln, = ax_disp.plot([], [], color=TRACK_COLORS[i], lw=1.2,
                               alpha=0.85, label=f"track {i+1}")
            disp_lines.append(ln)
        ax_disp.legend(loc="upper left", fontsize=8,
                       facecolor="#16213e", edgecolor="#333",
                       labelcolor="white")

        # ── Belt (row 2, only if enabled) ──
        if use_belt:
            ax_belt = fig.add_subplot(gs[2, :])
        else:
            ax_belt = None
    else:
        # Belt-only mode
        ax_bpm_b = fig.add_subplot(gs[0, :])
        ax_bpm_b.set_facecolor("#16213e")
        ax_bpm_b.set_xticks([]); ax_bpm_b.set_yticks([])
        for sp in ax_bpm_b.spines.values():
            sp.set_color("#333")
        belt_bpm_txt = ax_bpm_b.text(0.5, 0.55, "--", transform=ax_bpm_b.transAxes,
                                      fontsize=42, fontweight="bold", color="#ff79c6",
                                      ha="center", va="center", family="monospace")
        ax_bpm_b.text(0.5, 0.10, "belt BPM", transform=ax_bpm_b.transAxes,
                      fontsize=12, color="#aaa", ha="center", va="center")
        bpm_texts = [belt_bpm_txt]
        ax_belt = fig.add_subplot(gs[1, :])
        txt_status = None

    if ax_belt is not None:
        style_ax(ax_belt, "belt force waveform", "force (N)")
        line_belt, = ax_belt.plot([], [], color="#ff79c6", lw=1.2)
        cursor_belt = ax_belt.axvline(0, color="#ff6361", lw=1)
        ax_belt.set_xlim(0, BELT_HISTORY_S)
        ax_belt.set_ylim(0, 15)
        ax_belt.set_xlabel("time (s)", color="white")

    # ── 6. Animation callback ──────────────────────────────────────
    def update(_):
        nonlocal frame_count, bg_virtual_saved, bg_noise_std, track_mgr
        nonlocal range_profile_db, ra_map_db, belt_bpm, last_belt_bpm_time
        nonlocal last_rfft_tx0, last_rfft_tx2

        # ════ RADAR ════
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

                # Separate the 3 TX chirps
                tx0_chirps, tx1_chirps, tx2_chirps = separate_tx(frame)

                # Range FFT for each TX (averaged across the 1 chirp it has)
                rfft_tx0 = compute_range_fft(tx0_chirps, win_hann)  # (4, n_bins)
                rfft_tx1 = compute_range_fft(tx1_chirps, win_hann)  # (4, n_bins)
                rfft_tx2 = compute_range_fft(tx2_chirps, win_hann)  # (4, n_bins)

                last_rfft_tx0 = rfft_tx0
                last_rfft_tx2 = rfft_tx2

                # Azimuth virtual array (TX0+TX2) for AoA: 8 elements at λ/2
                azimuth_virtual = np.concatenate([rfft_tx0, rfft_tx2], axis=0)

                # Full 12-element virtual (TX0+TX1+TX2) for max-SNR range profile.
                # TX1 is elevation-offset so we use it only for range/SNR, not AoA.
                virtual = np.concatenate([rfft_tx0, rfft_tx1, rfft_tx2], axis=0)

                # 1D range profile (averaged across virtual elements)
                rfft_avg = virtual.mean(axis=0)  # (n_bins,)

                frame_count += 1
                t_rel = wall_now - t_epoch

                # ── Calibration ──
                if frame_count <= BG_FRAMES:
                    bg_frames_buf.append(virtual.copy())
                    pct = frame_count / BG_FRAMES * 100
                    secs_left = (BG_FRAMES - frame_count) * FRAME_PERIOD_S
                    if txt_status:
                        txt_status.set_text(
                            f"calibrating… {pct:.0f}% ({secs_left:.0f}s)")
                    continue

                if bg_virtual_saved is None:
                    # Stack: (BG_FRAMES, 8, n_bins)
                    all_v = np.array(bg_frames_buf)
                    bg_virtual_saved = np.mean(all_v, axis=0)  # (12, n_bins)

                    # Noise std from 1D averaged profile
                    avg_all = all_v.mean(axis=1)  # (BG_FRAMES, n_bins)
                    bg_noise_std = np.std(np.abs(avg_all), axis=0)

                    bg_frames_buf.clear()
                    print(f"[RADAR] background done ({BG_FRAMES} frames, "
                          f"12 virtual elements, 8 used for azimuth)")

                # Background subtraction on full 12-element virtual array
                diff_virtual = virtual - bg_virtual_saved  # (12, n_bins)

                # Adaptive background update — slow EMA removes new clutter
                # (object placement, drift) that wasn't in calibration.
                # Breathing oscillations pass through because they're faster
                # than the 10-second EMA time constant.
                #
                # We FREEZE the update at range bins near confirmed tracks so
                # that tracked people don't slowly get absorbed into the
                # background (which would cause them to become invisible).
                if ADAPTIVE_BG_ENABLED:
                    # Build a per-bin alpha mask
                    alpha_mask = np.full(n_bins, ADAPTIVE_BG_ALPHA, dtype=np.float32)
                    if track_mgr is not None:
                        for trk in track_mgr.confirmed_tracks:
                            bi = int(round(trk.bin_position))
                            lo = max(0, bi - 4)
                            hi = min(n_bins, bi + 5)
                            alpha_mask[lo:hi] = 0.0   # freeze around track
                    # Broadcast across virtual dim: (12, n_bins)
                    am = alpha_mask[np.newaxis, :]
                    bg_virtual_saved = ((1 - am) * bg_virtual_saved + am * virtual)

                # 1D range profile from all 12 elements (max SNR)
                diff_avg = diff_virtual.mean(axis=0)
                mag = np.abs(diff_avg)
                mag_db = 20 * np.log10(mag + 1e-6)
                range_profile_db = mag_db[keep_mask]

                # 2D range-angle map from azimuth subarray only (TX0+TX2 = elements 0-3 and 8-11)
                azimuth_diff = np.concatenate([diff_virtual[0:4],
                                                diff_virtual[8:12]], axis=0)
                ra_full = compute_range_angle_map(azimuth_diff)  # (AOA_FFT_SIZE, n_bins)
                ra_map_db = ra_full[angle_mask][:, keep_mask]

                # Track manager init
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
                    if txt_status:
                        txt_status.set_text("detecting targets…")
                    print("[RADAR] track manager ready (MIMO + 2D)")

                track_mgr.step(mag, diff_avg, t_rel)

                # AoA per track
                if last_rfft_tx0 is not None:
                    for trk in track_mgr.confirmed_tracks:
                        bi = int(round(trk.bin_position))
                        track_angles[trk.track_id] = estimate_aoa(
                            last_rfft_tx0, last_rfft_tx2, bi)

            # ── Update plots ──
            if track_mgr is not None:
                # Fan-shaped radar display
                pcm_ra.set_array(ra_map_db.ravel())
                p5, p95 = np.percentile(ra_map_db, [5, 95])
                pcm_ra.set_clim(vmin=max(p5, -50), vmax=max(p95 + 5, p5 + 10))

                confirmed = track_mgr.confirmed_tracks
                n_conf = len(confirmed)

                # Overlay: validated tracks → filled colored arc patch on
                #          the heatmap (matching their BPM color).
                #          Pending tracks → small grey scatter dot.

                # Clear previous patches
                for p in track_patches:
                    p.remove()
                track_patches.clear()

                if n_conf > 0:
                    # Pending markers as scatter dots
                    pend_x, pend_y = [], []

                    for t in confirmed:
                        a = track_angles.get(t.track_id, 0)
                        r = t.range_m

                        if t.breathing_validated:
                            # Draw an annular wedge in polar coords that lands
                            # at (r, a) — makes the person's location on the
                            # radar the SAME color as their BPM readout.
                            from matplotlib.patches import Wedge
                            # Angular half-width of the marker (degrees)
                            half_a = 6.0
                            # Radial half-width (meters)
                            half_r = 0.15
                            r_in  = max(r - half_r, 0.05)
                            r_out = r + half_r
                            # Wedge angles are measured from +x axis counterclockwise
                            # but our fan uses angle=0 as forward (+y).
                            # In cartesian: point at angle θ (measured from y) is
                            # at (r sinθ, r cosθ). Wedge theta_start is from +x.
                            # Our angle α (from y, right = +) corresponds to
                            # matplotlib angle β = 90 - α (measured from +x, ccw).
                            beta = 90.0 - a
                            wedge = Wedge(
                                center=(0, 0),
                                r=r_out,
                                theta1=beta - half_a,
                                theta2=beta + half_a,
                                width=r_out - r_in,
                                facecolor=t.color,
                                edgecolor="white",
                                linewidth=1.5,
                                alpha=0.85,
                                zorder=5,
                            )
                            ax_ra.add_patch(wedge)
                            track_patches.append(wedge)
                        else:
                            pend_x.append(r * np.sin(np.radians(a)))
                            pend_y.append(r * np.cos(np.radians(a)))

                    if pend_x:
                        scatter_tracks.set_offsets(np.column_stack([pend_x, pend_y]))
                        scatter_tracks.set_facecolors(["#555555"] * len(pend_x))
                        scatter_tracks.set_edgecolors(["#888888"] * len(pend_x))
                        scatter_tracks.set_sizes([70] * len(pend_x))
                    else:
                        scatter_tracks.set_offsets(np.empty((0, 2)))
                else:
                    scatter_tracks.set_offsets(np.empty((0, 2)))

                if txt_status:
                    if n_conf == 0:
                        txt_status.set_text("detecting targets…")
                    else:
                        n_val = sum(1 for t in confirmed if t.breathing_validated)
                        n_pend = n_conf - n_val
                        parts = []
                        for t in confirmed:
                            a = track_angles.get(t.track_id, 0)
                            state = "✓" if t.breathing_validated else "…"
                            parts.append(f"{state}{t.range_m:.1f}m/{a:+.0f}°")
                        txt_status.set_text(
                            f"{n_val} breathing, {n_pend} pending: "
                            f"{', '.join(parts)}")

                # BPM readouts — only show BPM for validated tracks
                for i in range(MAX_TRACKS):
                    if i < n_conf:
                        t = confirmed[i]
                        if t.breathing_validated:
                            bpm_texts[i].set_color(t.color)
                            bpm_texts[i].set_text(f"{t.bpm:.0f}" if t.bpm > 0 else "--")
                        else:
                            bpm_texts[i].set_color("#555555")
                            bpm_texts[i].set_text("…")
                    else:
                        bpm_texts[i].set_text("--")
                        bpm_texts[i].set_color("#333333")

                # Displacement waveforms — only draw for validated tracks
                latest_t = 0
                any_data = False
                for i in range(MAX_TRACKS):
                    if (i < n_conf and confirmed[i].breathing_validated
                            and len(confirmed[i].disp_hist) > 1):
                        t = confirmed[i]
                        ts = np.array(t.time_hist)
                        ds = np.array(t.disp_hist)
                        disp_lines[i].set_data(ts, ds)
                        disp_lines[i].set_color(t.color)
                        latest_t = max(latest_t, ts[-1])
                        any_data = True
                    else:
                        disp_lines[i].set_data([], [])

                if any_data:
                    ax_disp.set_xlim(max(0, latest_t - HISTORY_S), latest_t + 0.5)
                    all_v = []
                    for i in range(min(n_conf, MAX_TRACKS)):
                        if confirmed[i].breathing_validated and confirmed[i].disp_hist:
                            all_v.extend(confirmed[i].disp_hist)
                    if all_v:
                        margin = max(float(np.max(np.abs(all_v))) * 1.3, 0.5)
                        ax_disp.set_ylim(-margin, margin)

        # ════ BELT ════
        if use_belt and ax_belt is not None:
            with belt_lock:
                if len(belt_data) < 2:
                    return
                snap = list(belt_data)

            belt_t = np.array([s[0] - t_epoch for s in snap])
            belt_f = np.array([s[1] for s in snap])

            if len(belt_f) > 30:
                dt_med = float(np.median(np.diff(belt_t)))
                bfs = 1.0 / dt_med if dt_med > 0 else BELT_NOMINAL_HZ
                if len(belt_f) > 27:
                    try:
                        sos_b = make_bandpass(BELT_BP_LO, BELT_BP_HI, bfs)
                        bf = sosfiltfilt(sos_b, belt_f - belt_f.mean())
                    except Exception:
                        bf = belt_f - belt_f.mean()
                else:
                    bf = belt_f - belt_f.mean()

                # Only recompute BPM every BPM_REFRESH_S seconds
                now = time.monotonic()
                if now - last_belt_bpm_time >= BPM_REFRESH_S:
                    wn = min(len(bf), int(BELT_BPM_WINDOW_S * bfs))
                    belt_bpm, _ = estimate_bpm(bf[-wn:], bfs)
                    last_belt_bpm_time = now

            # Belt BPM readout
            bt = bpm_texts[-1]
            if belt_bpm > 0:
                bt.set_text(f"{belt_bpm:.0f}")
                bt.set_color("#ff79c6" if 10 <= belt_bpm <= 24 else
                             "#feca57" if belt_bpm < 10 else "#ff6361")

            line_belt.set_data(belt_t, belt_f)
            ax_belt.set_xlim(max(0, belt_t[-1] - BELT_HISTORY_S), belt_t[-1] + 0.5)
            ax_belt.set_ylim(float(belt_f.min()) - 1, float(belt_f.max()) + 1)
            cursor_belt.set_xdata([belt_t[-1], belt_t[-1]])

    # ── 7. Launch ──
    ani = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    mode = "3TX/4RX MIMO"
    if use_belt:
        mode += " + belt"
    print(f"\n[MAIN] {mode} dashboard running\n")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    # ── 8. Cleanup ──
    if belt_reader:
        belt_reader.stop()
        belt_reader.join(timeout=3)
    if radar_src:
        radar_src.stop()
        radar_src.join(timeout=2)
    if godirect_ctx:
        godirect_ctx.quit()
    print("\n[DONE]")


if __name__ == "__main__":
    main()