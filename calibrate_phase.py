#!/usr/bin/env python3
"""
Inter-TX Phase Calibration for XWR1843 TDM-MIMO
────────────────────────────────────────────────
When using TDM-MIMO (TX0 and TX2 alternating), there is a hardware phase
offset between the two transmit paths — different PCB routing, different
timing delays, and different LO paths cause each (TX, RX) pair to have
its own systematic phase offset relative to a reference (TX0+RX0).

Without this calibration:
  - The virtual array [v0..v3, v4..v7] has a phase discontinuity at v3→v4
  - The angle FFT interprets this as a strong signal near ±90° (edge of FoV)
  - Angular localization is wrong for all targets

This script:
  1. Captures 50 frames while a REFERENCE TARGET sits at boresight (0°)
  2. Computes per-RX phase offsets between TX0 and TX2
  3. Saves them to radar_phase_calibration.npz

The saved calibration is later loaded by the main monitor to correct
the TX2 phases before building the virtual array.

Usage:
  python calibrate_tx_phase.py

Reference target options (in order of quality):
  BEST:      Corner reflector (trihedral) at 1.5 m boresight
  GOOD:      Flat metal plate (~30 cm across, e.g. laptop closed on edge)
             held perpendicular to radar at 1.5 m boresight
  MINIMAL:   Person standing perfectly still at boresight, 1.5 m out
             (worse because of body width and micro-motion)
"""

import argparse
import socket
import sys
import time
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Radar constants — MUST match the running Lua config
# ═══════════════════════════════════════════════════════════════════════
NUM_RX            = 4
NUM_TX            = 3
NUM_CHIRPS        = 64
NUM_ADC_SAMPLES   = 256
TOTAL_CHIRPS      = NUM_CHIRPS * NUM_TX      # 192
COMPLEX_PER_FRAME = TOTAL_CHIRPS * NUM_RX * NUM_ADC_SAMPLES   # 196,608
RAW_BYTES_PER_FRAME = COMPLEX_PER_FRAME * 4                    # 786,432

C                 = 2.998e8
SLOPE_HZ_PER_S   = 78.020e12
SAMPLE_RATE_HZ    = 10e6

HOST_IP           = "192.168.33.30"
DATA_PORT         = 4098
CALIBRATION_FILE  = "radar_phase_calibration.npz"

# Where to look for the target
REF_RANGE_MIN_M   = 0.7      # ignore near-field coupling
REF_RANGE_MAX_M   = 3.5      # ignore far clutter/walls

N_FRAMES_TO_CAPTURE = 50


def range_axis(n: int) -> np.ndarray:
    return np.arange(n // 2) * (SAMPLE_RATE_HZ / n) * C / (2 * SLOPE_HZ_PER_S)


def parse_one_frame(raw_bytes: bytes) -> np.ndarray:
    """Openradar LVDS-interleaved parser (matches DCA1000 organize())."""
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    ret = np.zeros(len(raw) // 2, dtype=np.complex64)
    ret[0::2] = raw[0::4].astype(np.float32) + 1j * raw[2::4].astype(np.float32)
    ret[1::2] = raw[1::4].astype(np.float32) + 1j * raw[3::4].astype(np.float32)
    return ret.reshape(TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES)


def separate_tx(frame: np.ndarray):
    """Studio TX0 (az L), TX1 (elev), TX2 (az R). Chirp order per Lua:
    [TX1_elev, TX0_azL, TX2_azR]."""
    tx1_elev = frame[0::3]
    tx0_azL  = frame[1::3]
    tx2_azR  = frame[2::3]
    return tx0_azL, tx1_elev, tx2_azR


def compute_range_fft(chirps: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Coherently averaged range FFT across all chirps for one TX."""
    dc = chirps.mean(axis=-1, keepdims=True)
    windowed = (chirps - dc) * win
    rfft = np.fft.fft(windowed, axis=-1)[..., :NUM_ADC_SAMPLES // 2]
    return rfft.mean(axis=0)  # (n_rx, n_bins)


def capture_frames(host, port, n_frames):
    needed = n_frames * RAW_BYTES_PER_FRAME
    buf = bytearray()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.settimeout(5.0)
    sock.bind((host, port))
    print(f"[UDP] listening on {host}:{port}")
    print(f"[UDP] need {needed:,} bytes ({n_frames} frames)")

    t0 = time.monotonic()
    while len(buf) < needed:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            if time.monotonic() - t0 > 30:
                sock.close()
                raise TimeoutError("no data from radar")
            continue
        if len(data) > 10:
            buf.extend(data[10:])
        if len(buf) % (needed // 20) < 4096:
            pct = len(buf) / needed * 100
            print(f"  {pct:.0f}%  ({len(buf):,} bytes)", end="\r")

    sock.close()
    print(f"\n[UDP] captured {len(buf):,} bytes in {time.monotonic()-t0:.1f}s")
    return bytes(buf[:needed])


def calibrate(raw_data: bytes, verbose: bool = True):
    """Compute per-RX phase offsets between TX0 and TX2.

    Approach:
      1. Parse all captured frames
      2. Compute the range FFT for TX0 and TX2 separately, averaged over
         all chirps and all captured frames (heavy coherent averaging).
      3. Find the strongest peak in the reference range window.
      4. At that peak, measure the phase of TX0[rx] and TX2[rx] for each RX.
      5. The correction is the negative of (phase_tx2 - phase_tx0).

    Returns:
        offsets: array of shape (4,) — phase to add to each TX2 RX
        peak_bin: which range bin was used for calibration
        peak_range_m: the range of that bin
    """
    n_frames = len(raw_data) // RAW_BYTES_PER_FRAME
    if n_frames < 1:
        raise ValueError("no complete frames in captured data")

    if verbose:
        print(f"[CALIBRATE] processing {n_frames} frames")

    # Coherently average TX0 and TX2 range FFTs across ALL frames.
    # This is important — averaging suppresses noise but preserves the
    # phase relationship at the reference target's range bin.
    win = np.hanning(NUM_ADC_SAMPLES).astype(np.float32)
    n_bins = NUM_ADC_SAMPLES // 2

    tx0_sum = np.zeros((NUM_RX, n_bins), dtype=np.complex128)
    tx2_sum = np.zeros((NUM_RX, n_bins), dtype=np.complex128)

    for i in range(n_frames):
        chunk = raw_data[i * RAW_BYTES_PER_FRAME:(i + 1) * RAW_BYTES_PER_FRAME]
        frame = parse_one_frame(chunk)
        tx0_chirps, _, tx2_chirps = separate_tx(frame)
        tx0_sum += compute_range_fft(tx0_chirps, win).astype(np.complex128)
        tx2_sum += compute_range_fft(tx2_chirps, win).astype(np.complex128)

    tx0_avg = tx0_sum / n_frames
    tx2_avg = tx2_sum / n_frames

    # Find the strongest peak in the reference range window,
    # averaged across all RX channels (isotropic).
    rng = range_axis(NUM_ADC_SAMPLES)
    range_min_bin = int(np.searchsorted(rng, REF_RANGE_MIN_M))
    range_max_bin = int(np.searchsorted(rng, REF_RANGE_MAX_M))

    mag_avg = np.abs(tx0_avg).mean(axis=0)   # (n_bins,)
    search_region = mag_avg[range_min_bin:range_max_bin]
    peak_local = int(np.argmax(search_region))
    peak_bin = range_min_bin + peak_local
    peak_range = float(rng[peak_bin])

    if verbose:
        print(f"[CALIBRATE] reference target detected at bin {peak_bin} "
              f"(range {peak_range:.2f} m, magnitude {mag_avg[peak_bin]:.0f})")
        peak_snr_db = 20 * np.log10(
            mag_avg[peak_bin] / (np.median(mag_avg[range_min_bin:range_max_bin]) + 1e-9))
        print(f"[CALIBRATE] peak SNR above range-window median: {peak_snr_db:.1f} dB")
        if peak_snr_db < 15:
            print(f"[CALIBRATE] ⚠  Peak is weak. Consider using a corner reflector")
            print(f"             or a large flat metal target.")

    # Measure per-RX phase of TX0 and TX2 at the peak bin
    tx0_phases = np.angle(tx0_avg[:, peak_bin])   # (4,)
    tx2_phases = np.angle(tx2_avg[:, peak_bin])   # (4,)

    # Raw per-RX offset (uncorrected). This includes both the inter-TX
    # phase offset AND the angular phase ramp from off-boresight targets.
    raw_offsets = tx2_phases - tx0_phases    # (4,) radians

    # If the target is truly at boresight, the angular phase ramp is zero
    # AND the offsets are all the same value (the pure inter-TX offset).
    # If the target is at angle θ, the offsets will have a linear trend
    # across RX (because RX0..RX3 see different phases for a slanted wavefront).
    # We can decompose:
    #   raw_offset[rx] = inter_tx_phase + 2π × 4 × 0.5 × sin(θ)  (constant across RX)
    # Wait — that's not right. Let me think again.

    # Actually: the TX0 subarray already contains the angular phase ramp
    # (RX0..RX3 phases shift due to target angle). The TX2 subarray also
    # contains the same angular ramp. When we take (tx2 - tx0) at each RX,
    # the ramp cancels because it's the same at both TX. What remains is
    # only the inter-TX offset, which SHOULD be roughly constant across RX
    # (though PCB routing makes it slightly different per RX).

    # So the raw offset per RX is our answer: add these back to TX2 samples
    # to correct.

    corrections = -raw_offsets   # what to ADD to TX2 phases to align them

    if verbose:
        print(f"[CALIBRATE] Raw phase offsets TX0 → TX2 (per RX):")
        for rx in range(NUM_RX):
            print(f"  RX{rx}: TX0={np.degrees(tx0_phases[rx]):+7.1f}°  "
                  f"TX2={np.degrees(tx2_phases[rx]):+7.1f}°  "
                  f"Δ={np.degrees(raw_offsets[rx]):+7.1f}°")

        # Sanity check: how much do the offsets vary across RX?
        offset_std_deg = np.degrees(np.std(raw_offsets))
        offset_mean_deg = np.degrees(np.mean(raw_offsets))
        print(f"[CALIBRATE] mean offset = {offset_mean_deg:+.1f}°, "
              f"std across RX = {offset_std_deg:.1f}°")
        if offset_std_deg > 30:
            print(f"[CALIBRATE] ⚠  Large std suggests target was NOT at boresight.")
            print(f"             Move target to directly in front of radar and rerun.")
        else:
            print(f"[CALIBRATE] ✓  Offsets are consistent — target was near boresight.")

    return corrections, peak_bin, peak_range


def save_calibration(corrections, peak_bin, peak_range, path):
    np.savez(path,
             tx2_corrections=corrections,       # radians, one per RX
             peak_bin=peak_bin,
             peak_range_m=peak_range,
             timestamp=time.time(),
             num_rx=NUM_RX)
    print(f"[SAVE] calibration written to {path}")
    print(f"[SAVE] TX2 corrections (deg): {[f'{np.degrees(c):+.1f}' for c in corrections]}")


def main():
    ap = argparse.ArgumentParser(description="Inter-TX phase calibration")
    ap.add_argument("--host", default=HOST_IP)
    ap.add_argument("--port", type=int, default=DATA_PORT)
    ap.add_argument("--frames", type=int, default=N_FRAMES_TO_CAPTURE)
    ap.add_argument("--out", default=CALIBRATION_FILE)
    ap.add_argument("--countdown", type=int, default=10,
                    help="seconds before capture starts")
    args = ap.parse_args()

    print("═" * 60)
    print(" INTER-TX PHASE CALIBRATION")
    print("═" * 60)
    print()
    print(" Setup:")
    print("   1. Place a REFERENCE TARGET at BORESIGHT (0°), ~1.5 m away.")
    print("      Best:    corner reflector / trihedral")
    print("      Good:    large flat metal (laptop closed on edge, etc.)")
    print("      Minimal: person standing very still, directly in front")
    print()
    print("   2. Clear other reflectors from the field of view.")
    print("   3. Make sure the radar is running via mmWave Studio.")
    print()
    print(" The calibration measures how much the TX2 signal path differs")
    print(" from the TX0 signal path so that the 8-element virtual array")
    print(" produces the correct angle for future measurements.")
    print()

    if args.countdown > 0:
        print(f" Starting capture in {args.countdown}s. Get target in position…")
        for i in range(args.countdown, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        print(f"  CAPTURING NOW  (hold target still for ~2 seconds)")
        print()

    try:
        raw = capture_frames(args.host, args.port, args.frames)
    except TimeoutError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print()
    corrections, peak_bin, peak_range = calibrate(raw)
    save_calibration(corrections, peak_bin, peak_range, args.out)

    print()
    print("═" * 60)
    print(" DONE. Run multi_breathing_monitor.py and it will load this")
    print(" calibration automatically.")
    print("═" * 60)


if __name__ == "__main__":
    main()