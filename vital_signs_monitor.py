#!/usr/bin/env python3
"""
Real-time breathing-rate monitor for TI mmWave radar + DCA1000.

Combines the full pipeline in one script:
  1. UDP listener (background thread) captures raw ADC packets
  2. Binary parser assembles IQ frames on the fly
  3. Range FFT  →  phase extraction  →  bandpass  →  BPM estimation
  4. Live matplotlib display with:
       • range profile (top-left)
       • chest-displacement waveform (bottom)
       • large BPM readout (top-right)

Usage:
  sudo python realtime_breathing_monitor.py
  sudo python realtime_breathing_monitor.py --host 192.168.33.30 --port 4098
  sudo python realtime_breathing_monitor.py --demo   # synthetic data for testing
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

matplotlib.use("TkAgg")  # interactive backend — change to "Qt5Agg" if preferred
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt

# ═══════════════════════════════════════════════════════════════════════
# Radar / capture constants  (must match your .json / DCA1000 config)
# ═══════════════════════════════════════════════════════════════════════
NUM_RX            = 1
NUM_TX            = 1
NUM_ADC_SAMPLES   = 540
NUM_CHIRPS        = 128
FRAME_PERIOD_S    = 39.99975e-3          # ~25 Hz frame rate
FPS_HZ            = 1.0 / FRAME_PERIOD_S

C                 = 2.998e8
SLOPE_HZ_PER_S   = 70.006e12
SAMPLE_RATE_HZ   = 10e6
F0_HZ             = 77.0000000238419e9
LAMBDA_M          = C / F0_HZ

# Per frame: 128 chirps × 1 TX × 1 RX × 540 samples = 69 120 complex samples
# Each complex sample occupies 4 int16 in the raw stream (I, 0-pad, Q, 0-pad) = 8 bytes
COMPLEX_PER_FRAME = NUM_CHIRPS * NUM_TX * NUM_RX * NUM_ADC_SAMPLES
RAW_BYTES_PER_FRAME = COMPLEX_PER_FRAME * 8  # 552 960 bytes

# Processing
BG_FRAMES         = 50        # frames used as static background
HISTORY_S         = 30        # seconds of displacement history to keep
MAX_HISTORY       = int(HISTORY_S * FPS_HZ)
BP_LO, BP_HZ     = 0.1, 0.6  # breathing band (Hz)
BPM_WINDOW_S      = 15        # seconds of data for spectral BPM estimate
XMAX_M            = 2.0       # range axis limit for the plot

# UDP
HOST_IP           = "192.168.33.30"
DATA_PORT         = 4098


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def range_axis(n: int) -> np.ndarray:
    """Metres per FFT bin."""
    return np.arange(n // 2) * (SAMPLE_RATE_HZ / n) * C / (2 * SLOPE_HZ_PER_S)


def make_bandpass(lo, hi, fs, order=4):
    """Butterworth bandpass as second-order sections."""
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def estimate_bpm(signal_seg, fs):
    """Spectral peak in the breathing band → BPM."""
    if len(signal_seg) < 64:
        return 0.0, 0.0
    sig = signal_seg - signal_seg.mean()
    f = np.fft.rfftfreq(len(sig), d=1 / fs)
    mag = np.abs(np.fft.rfft(sig))
    band = (f >= BP_LO) & (f <= BP_HZ)
    if not band.any():
        return 0.0, 0.0
    peak_f = float(f[band][np.argmax(mag[band])])
    return peak_f * 60.0, peak_f


# ═══════════════════════════════════════════════════════════════════════
# UDP capture thread
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
        self.packets = 0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.settimeout(1.0)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            print(f"ERROR: Cannot bind {self.host}:{self.port} — {e}")
            print(f"  Try: sudo python {sys.argv[0]}")
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
                payload = data[10:]  # strip 10-byte DCA1000 header
                with self.lock:
                    self.buf.extend(payload)
                self.total_bytes += len(payload)
                self.packets += 1
        sock.close()

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════
# Synthetic demo source (no radar needed)
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
        print(f"[DEMO] generating synthetic frames at {FPS_HZ:.0f} Hz, "
              f"simulated breathing = {self.bpm:.0f} BPM")
        rng = range_axis(NUM_ADC_SAMPLES)
        target_bin = int(np.searchsorted(rng, 0.5))  # ~0.5 m target
        breath_f = self.bpm / 60.0

        while self.running:
            t = self.frame_idx * FRAME_PERIOD_S
            # Simulate: strong reflector at target_bin with breathing phase modulation
            frame_iq = np.zeros((NUM_CHIRPS, NUM_RX, NUM_ADC_SAMPLES), dtype=np.complex64)
            phase_shift = 0.005 * np.sin(2 * np.pi * breath_f * t)  # small phase
            for c in range(NUM_CHIRPS):
                tone = np.zeros(NUM_ADC_SAMPLES, dtype=np.complex64)
                # place energy at target_bin via a sinusoid at the correct freq
                freq_bin = target_bin / NUM_ADC_SAMPLES
                n = np.arange(NUM_ADC_SAMPLES)
                tone = 1000 * np.exp(1j * (2 * np.pi * freq_bin * n + phase_shift))
                tone += (np.random.randn(NUM_ADC_SAMPLES) +
                         1j * np.random.randn(NUM_ADC_SAMPLES)) * 5
                frame_iq[c, 0, :] = tone

            # Flatten to I, Q interleaved int16 with zero-pad
            flat = frame_iq.reshape(-1)
            i_vals = np.round(flat.real).astype(np.int16)
            q_vals = np.round(flat.imag).astype(np.int16)
            # interleave: I, 0, Q, 0
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
# Frame parser  (same logic as binparser.py, one frame at a time)
# ═══════════════════════════════════════════════════════════════════════

def parse_one_frame(raw_bytes: bytes) -> np.ndarray:
    """
    Parse RAW_BYTES_PER_FRAME bytes into (NUM_CHIRPS, NUM_RX, NUM_ADC_SAMPLES)
    complex64 array.
    """
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    samples = raw[0::2]  # drop zero-pad int16s
    iq = samples[0::2].astype(np.float32) + 1j * samples[1::2].astype(np.float32)
    return iq.reshape(NUM_CHIRPS * NUM_TX, NUM_RX, NUM_ADC_SAMPLES)


# ═══════════════════════════════════════════════════════════════════════
# Main — real-time plot
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Real-time breathing monitor")
    ap.add_argument("--host", default=HOST_IP)
    ap.add_argument("--port", type=int, default=DATA_PORT)
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic data (no radar hardware needed)")
    ap.add_argument("--bpm", type=float, default=15.0,
                    help="simulated breathing rate for --demo mode")
    args = ap.parse_args()

    # Shared buffer + lock
    raw_buf = bytearray()
    buf_lock = threading.Lock()

    if args.demo:
        src = DemoSource(raw_buf, buf_lock, bpm=args.bpm)
    else:
        src = UDPCapture(args.host, args.port, raw_buf, buf_lock)
    src.start()

    # ---- processing state ----
    rng = range_axis(NUM_ADC_SAMPLES)
    n_bins = NUM_ADC_SAMPLES // 2
    keep_mask = rng <= XMAX_M
    rng_plot = rng[keep_mask]

    win = np.hanning(NUM_ADC_SAMPLES).astype(np.float32)
    sos_bp = make_bandpass(BP_LO, BP_HZ, FPS_HZ)

    frame_count = 0
    bg_accum = np.zeros((n_bins,), dtype=np.complex128)  # running bg sum

    phase_hist = collections.deque(maxlen=MAX_HISTORY)
    disp_hist = collections.deque(maxlen=MAX_HISTORY)
    time_hist = collections.deque(maxlen=MAX_HISTORY)
    range_profile_db = np.full(keep_mask.sum(), -60.0)

    target_bin = None  # auto-detected after BG_FRAMES
    current_bpm = 0.0
    prev_phase = 0.0  # for unwrap

    # ---- figure layout ----
    fig = plt.figure(figsize=(11, 6.5))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1],
                  hspace=0.35, wspace=0.25)

    # top-left: range profile
    ax_rng = fig.add_subplot(gs[0, 0])
    ax_rng.set_facecolor("#16213e")
    line_rng, = ax_rng.plot(rng_plot, range_profile_db, color="#0ff", lw=1.4)
    vline_target = ax_rng.axvline(0.5, color="#ff6361", ls="--", lw=0.9, alpha=0.7)
    ax_rng.set_xlim(0, XMAX_M)
    ax_rng.set_ylim(-60, 40)
    ax_rng.set_xlabel("range (m)", color="white")
    ax_rng.set_ylabel("|H − bg|  (dB)", color="white")
    ax_rng.set_title("range profile", color="white", fontsize=11)
    ax_rng.tick_params(colors="white")
    ax_rng.grid(alpha=0.15, color="white")

    # top-right: big BPM readout
    ax_bpm = fig.add_subplot(gs[0, 1])
    ax_bpm.set_facecolor("#16213e")
    ax_bpm.set_xticks([])
    ax_bpm.set_yticks([])
    txt_bpm = ax_bpm.text(0.5, 0.55, "--", transform=ax_bpm.transAxes,
                          fontsize=64, fontweight="bold", color="#00ff99",
                          ha="center", va="center", family="monospace")
    ax_bpm.text(0.5, 0.15, "BPM", transform=ax_bpm.transAxes,
                fontsize=18, color="#aaaaaa", ha="center", va="center")
    txt_status = ax_bpm.text(0.5, 0.88, "calibrating…", transform=ax_bpm.transAxes,
                             fontsize=10, color="#ff9f43", ha="center", va="center")

    # bottom: displacement waveform (full width)
    ax_disp = fig.add_subplot(gs[1, :])
    ax_disp.set_facecolor("#16213e")
    line_disp, = ax_disp.plot([], [], color="#feca57", lw=1.2)
    cursor_disp = ax_disp.axvline(0, color="#ff6361", lw=1)
    ax_disp.set_xlim(0, HISTORY_S)
    ax_disp.set_ylim(-2, 2)
    ax_disp.set_xlabel("time (s)", color="white")
    ax_disp.set_ylabel("chest displacement (mm)", color="white")
    ax_disp.set_title("filtered displacement  (0.1 – 0.6 Hz)", color="white",
                       fontsize=11)
    ax_disp.tick_params(colors="white")
    ax_disp.grid(alpha=0.15, color="white")

    for spine in [*ax_rng.spines.values(), *ax_bpm.spines.values(),
                  *ax_disp.spines.values()]:
        spine.set_color("#333")

    # ── animation callback ──────────────────────────────────────────
    def update(_frame_unused):
        nonlocal frame_count, bg_accum, target_bin, current_bpm, prev_phase
        nonlocal range_profile_db

        # Pull complete frames from the shared buffer
        frames_this_tick = []
        with buf_lock:
            while len(raw_buf) >= RAW_BYTES_PER_FRAME:
                chunk = bytes(raw_buf[:RAW_BYTES_PER_FRAME])
                del raw_buf[:RAW_BYTES_PER_FRAME]
                frames_this_tick.append(chunk)

        if not frames_this_tick:
            return line_rng, txt_bpm, line_disp, cursor_disp, txt_status

        for raw_chunk in frames_this_tick:
            frame = parse_one_frame(raw_chunk)  # (chirps, rx, samples)

            # --- range FFT (Hann + DC removal, coherent avg across chirps) ---
            dc = frame.mean(axis=-1, keepdims=True)
            windowed = (frame - dc) * win
            rfft = np.fft.fft(windowed, axis=-1)[..., :n_bins]
            rfft_avg = rfft.mean(axis=0).squeeze()  # (n_bins,) complex

            frame_count += 1
            t_now = frame_count * FRAME_PERIOD_S

            # --- background accumulation ---
            if frame_count <= BG_FRAMES:
                bg_accum += rfft_avg
                pct = frame_count / BG_FRAMES * 100
                txt_status.set_text(f"calibrating… {pct:.0f}%")
                continue

            if frame_count == BG_FRAMES + 1:
                bg = bg_accum / BG_FRAMES
                # auto-detect target bin (strongest peak 0.2–1.5 m)
                mag_avg = np.abs(bg)
                lo = max(1, int(np.searchsorted(rng, 0.2)))
                hi = int(np.searchsorted(rng, 1.5))
                target_bin = lo + int(np.argmax(mag_avg[lo:hi]))
                vline_target.set_xdata([rng[target_bin], rng[target_bin]])
                txt_status.set_text(f"tracking @ {rng[target_bin]:.2f} m")
                print(f"[PROC] bg done — target bin {target_bin} "
                      f"({rng[target_bin]:.3f} m)")

            bg = bg_accum / BG_FRAMES

            # --- range profile (dB, bg-subtracted) ---
            mag = np.abs(rfft_avg - bg)
            mag_db = 20 * np.log10(mag + 1e-6)
            range_profile_db = mag_db[keep_mask]

            # --- phase at target bin → displacement ---
            z = rfft_avg[target_bin]
            ang = np.angle(z)
            # manual unwrap relative to previous sample
            diff = ang - prev_phase
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            prev_phase = prev_phase + diff
            unwrapped = prev_phase

            phase_hist.append(unwrapped)
            time_hist.append(t_now)

            # convert phase to displacement (mm) with linear detrend
            if len(phase_hist) > 10:
                ph = np.array(phase_hist)
                ts = np.array(time_hist)
                a, b = np.polyfit(ts, ph, 1)
                ph_dt = ph - (a * ts + b)
                disp_mm = -LAMBDA_M * ph_dt / (4 * np.pi) * 1000.0

                # bandpass filter
                if len(disp_mm) > 27:  # must be strictly > padlen (27)
                    disp_filt = sosfiltfilt(sos_bp, disp_mm)
                else:
                    disp_filt = disp_mm

                disp_hist.clear()
                disp_hist.extend(disp_filt)

                # BPM estimate from the latest window
                win_samples = min(len(disp_filt),
                                  int(BPM_WINDOW_S * FPS_HZ))
                current_bpm, _ = estimate_bpm(disp_filt[-win_samples:], FPS_HZ)

        # ── update plots (only after background is done) ──
        if frame_count <= BG_FRAMES:
            return line_rng, txt_bpm, line_disp, cursor_disp, txt_status

        # range profile
        line_rng.set_ydata(range_profile_db)
        lo_db = float(np.min(range_profile_db)) - 3
        hi_db = float(np.max(range_profile_db)) + 3
        ax_rng.set_ylim(lo_db, hi_db)

        # BPM number
        if current_bpm > 0:
            txt_bpm.set_text(f"{current_bpm:.0f}")
            # colour: green=normal, yellow=low, red=high
            if 10 <= current_bpm <= 24:
                txt_bpm.set_color("#00ff99")
            elif current_bpm < 10:
                txt_bpm.set_color("#feca57")
            else:
                txt_bpm.set_color("#ff6361")

        # displacement waveform
        if len(disp_hist) > 1:
            ts_arr = np.array(time_hist)
            ds_arr = np.array(disp_hist)
            line_disp.set_data(ts_arr, ds_arr)
            ax_disp.set_xlim(max(0, ts_arr[-1] - HISTORY_S), ts_arr[-1] + 0.5)
            margin = max(float(np.abs(ds_arr).max()) * 1.3, 0.5)
            ax_disp.set_ylim(-margin, margin)
            cursor_disp.set_xdata([ts_arr[-1], ts_arr[-1]])

        return line_rng, txt_bpm, line_disp, cursor_disp, txt_status

    ani = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    plt.show()

    src.stop()
    src.join(timeout=2)
    print("\n[DONE]")


if __name__ == "__main__":
    main()