#!/usr/bin/env python3
"""
MIMO Data Diagnostic — checks if 2TX/4RX data is actually present
─────────────────────────────────────────────────────────────────
Captures a few raw frames from the DCA1000 and runs checks:
  1. Are all 4 RX channels distinct?
  2. Are TX0 and TX2 chirps different?
  3. What's the energy per RX channel?
  4. What's the cross-correlation between RX pairs?
  5. Does the data layout match what we expect?

Usage:
  python mimo_diagnostic.py
  python mimo_diagnostic.py --host 192.168.33.30 --port 4098
  python mimo_diagnostic.py --file raw_capture.bin   # from a saved file
"""

import argparse
import socket
import sys
import time
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Config — must match your mmWaveStudio settings
# ═══════════════════════════════════════════════════════════════════════
NUM_RX           = 4
NUM_TX           = 3       # TX0 + TX1 + TX2
NUM_CHIRPS       = 64      # chirp loops per TX
NUM_ADC_SAMPLES  = 256
TOTAL_CHIRPS     = NUM_CHIRPS * NUM_TX  # 192

# Complex1x format: 4 bytes per complex sample (I + Q as int16)
COMPLEX_PER_FRAME = TOTAL_CHIRPS * NUM_RX * NUM_ADC_SAMPLES  # 196,608
RAW_BYTES_PER_FRAME = COMPLEX_PER_FRAME * 4                  # 786,432

HOST_IP  = "192.168.33.30"
DATA_PORT = 4098
NUM_FRAMES_TO_CAPTURE = 5


def capture_raw_data(host, port, n_frames):
    """Capture n_frames worth of raw data from DCA1000 UDP."""
    needed = n_frames * RAW_BYTES_PER_FRAME
    buf = bytearray()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.settimeout(5.0)

    try:
        sock.bind((host, port))
    except OSError as e:
        print(f"ERROR: Cannot bind {host}:{port} — {e}")
        sys.exit(1)

    print(f"[UDP] listening on {host}:{port}")
    print(f"[UDP] need {needed:,} bytes ({n_frames} frames × {RAW_BYTES_PER_FRAME:,} bytes/frame)")
    print(f"[UDP] waiting for data… (start the radar if not running)")

    packet_count = 0
    t0 = time.monotonic()

    while len(buf) < needed:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            elapsed = time.monotonic() - t0
            if elapsed > 30:
                print(f"\nERROR: timeout after 30s. Got {len(buf):,} bytes "
                      f"({len(buf)/RAW_BYTES_PER_FRAME:.1f} frames)")
                break
            continue

        if len(data) > 10:
            buf.extend(data[10:])  # strip DCA1000 header
            packet_count += 1

            if packet_count % 200 == 0:
                pct = len(buf) / needed * 100
                print(f"  {pct:.0f}%  ({len(buf):,} / {needed:,} bytes, "
                      f"{packet_count} packets)", end="\r")

    sock.close()
    elapsed = time.monotonic() - t0
    print(f"\n[UDP] captured {len(buf):,} bytes in {elapsed:.1f}s "
          f"({packet_count} packets)")
    return bytes(buf)


def parse_frame(raw_bytes):
    """Parse one frame into (TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES) complex.

    Complex1x LVDS-interleaved format (openradar convention). Every 4
    consecutive int16s contain the real and imaginary parts of TWO
    complex samples:
        raw[0], raw[1] → real parts of samples 0 and 1
        raw[2], raw[3] → imaginary parts of samples 0 and 1
    """
    raw = np.frombuffer(raw_bytes[:RAW_BYTES_PER_FRAME], dtype=np.int16)
    ret = np.zeros(len(raw) // 2, dtype=np.complex64)
    ret[0::2] = raw[0::4].astype(np.float32) + 1j * raw[2::4].astype(np.float32)
    ret[1::2] = raw[1::4].astype(np.float32) + 1j * raw[3::4].astype(np.float32)
    return ret.reshape(TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES)


def run_diagnostics(raw_data):
    """Run all checks on the captured data."""

    n_complete = len(raw_data) // RAW_BYTES_PER_FRAME
    if n_complete == 0:
        print("\nERROR: not enough data for even one complete frame")
        print(f"  Got {len(raw_data):,} bytes, need {RAW_BYTES_PER_FRAME:,}")

        # Check if it matches 1TX/1RX frame size
        old_frame = 128 * 1 * 1 * 540 * 8  # 552,960
        n_old = len(raw_data) // old_frame
        if n_old > 0:
            print(f"\n  ⚠ Data DOES match 1TX/1RX frame size ({old_frame:,} bytes)")
            print(f"    Got {n_old} complete 1TX/1RX frames")
            print(f"    → The radar is likely still in 1TX/1RX mode!")
            print(f"    → Check that chirp 0 and chirp 1 are configured correctly")
            print(f"    → Check that End Chirp TX = 1 in the Frame section")
        return

    print(f"\n{'═'*60}")
    print(f" MIMO DIAGNOSTIC REPORT — {n_complete} frames captured")
    print(f"{'═'*60}")

    # Parse first frame for detailed analysis
    frame = parse_frame(raw_data)
    print(f"\n Frame shape: {frame.shape} "
          f"(expected ({TOTAL_CHIRPS}, {NUM_RX}, {NUM_ADC_SAMPLES}))")

    # ── Check 1: Raw data stats ──
    print(f"\n── Check 1: Raw data statistics ──")
    for rx in range(NUM_RX):
        rx_data = frame[:, rx, :]
        rms = float(np.sqrt(np.mean(np.abs(rx_data)**2)))
        max_val = float(np.max(np.abs(rx_data)))
        print(f"  RX{rx}: RMS={rms:.1f}  max={max_val:.1f}  "
              f"{'✓ active' if rms > 1.0 else '✗ DEAD/ZERO'}")

    # ── Check 2: Are RX channels distinct? ──
    print(f"\n── Check 2: RX channel cross-correlation ──")
    print(f"  (values near 1.0 = identical data, near 0.0 = independent)")

    # Use the range FFT of the first chirp for comparison
    fft_per_rx = []
    for rx in range(NUM_RX):
        fft_data = np.fft.fft(frame[0, rx, :])
        fft_per_rx.append(fft_data)

    for i in range(NUM_RX):
        for j in range(i + 1, NUM_RX):
            a = fft_per_rx[i]
            b = fft_per_rx[j]
            # Normalized cross-correlation
            corr = float(np.abs(np.sum(a * np.conj(b))) /
                        (np.sqrt(np.sum(np.abs(a)**2)) * np.sqrt(np.sum(np.abs(b)**2))))
            status = "⚠ IDENTICAL" if corr > 0.99 else "✓ distinct" if corr < 0.95 else "~ similar"
            print(f"  RX{i} vs RX{j}: correlation = {corr:.4f}  {status}")

    # ── Check 3: Are TX0 and TX2 chirps different? ──
    print(f"\n── Check 3: TX0 vs TX2 chirp separation ──")
    # Chirp order from Lua script:
    #   Chirp 0 → TX1 (elevated)
    #   Chirp 1 → TX0
    #   Chirp 2 → TX2
    tx1_chirps = frame[0::3]   # chirps 0, 3, 6, ... = TX1
    tx0_chirps = frame[1::3]   # chirps 1, 4, 7, ... = TX0
    tx2_chirps = frame[2::3]   # chirps 2, 5, 8, ... = TX2
    print(f"  TX0 chirps shape: {tx0_chirps.shape}")
    print(f"  TX1 chirps shape: {tx1_chirps.shape}  (elevation antenna)")
    print(f"  TX2 chirps shape: {tx2_chirps.shape}")

    tx0_rms = float(np.sqrt(np.mean(np.abs(tx0_chirps)**2)))
    tx2_rms = float(np.sqrt(np.mean(np.abs(tx2_chirps)**2)))
    print(f"  TX0 RMS: {tx0_rms:.1f}")
    print(f"  TX2 RMS: {tx2_rms:.1f}")

    if tx2_rms < 0.1:
        print(f"  ✗ TX2 appears DEAD — no signal on TX2 chirps")
        print(f"    → Check chirp 1 has TX2 Enable = 1 in ChirpManager")
    elif abs(tx0_rms - tx2_rms) / max(tx0_rms, tx2_rms) < 0.01:
        print(f"  ~ TX0 and TX2 have very similar RMS — could be the same TX")
    else:
        print(f"  ✓ TX0 and TX2 have different signal levels")

    # Compare TX0 and TX2 on the same RX to see if they're actually different
    for rx in range(NUM_RX):
        a = tx0_chirps[0, rx, :]  # first TX0 chirp, this RX
        b = tx2_chirps[0, rx, :]  # first TX2 chirp, this RX
        corr = float(np.abs(np.sum(a * np.conj(b))) /
                    (np.sqrt(np.sum(np.abs(a)**2)) * np.sqrt(np.sum(np.abs(b)**2)) + 1e-12))
        status = "⚠ IDENTICAL" if corr > 0.99 else "✓ distinct"
        print(f"  TX0 vs TX2 on RX{rx}: correlation = {corr:.4f}  {status}")

    # ── Check 4: Range profile per RX per TX ──
    print(f"\n── Check 4: Range FFT peak per RX × TX ──")
    print(f"  {'':>8s}", end="")
    for rx in range(NUM_RX):
        print(f"  {'RX'+str(rx):>10s}", end="")
    print()

    for tx_label, chirps in [("TX0", tx0_chirps), ("TX2", tx2_chirps)]:
        print(f"  {tx_label:>8s}", end="")
        for rx in range(NUM_RX):
            # Average range FFT across chirps for this TX/RX
            avg_fft = np.mean(np.fft.fft(chirps[:, rx, :], axis=-1), axis=0)
            mag = np.abs(avg_fft[:NUM_ADC_SAMPLES // 2])
            # Skip bins 0-10 (~0.75m) to avoid TX-to-RX antenna coupling
            peak_bin = int(np.argmax(mag[10:])) + 10
            peak_val = float(mag[peak_bin])
            print(f"  bin{peak_bin:3d}={peak_val:5.0f}", end="")
        print()

    # ── Check 5: Phase across virtual array at multiple ranges ──
    print(f"\n── Check 5: Phase ramp scan across range bins ──")
    print(f"  (looking for a strong target with clean phase ramp)")

    all_fft = np.fft.fft(frame[:, :, :], axis=-1)[..., :NUM_ADC_SAMPLES // 2]
    avg_all_db = 20 * np.log10(np.abs(all_fft.mean(axis=(0, 1))) + 1e-6)

    # Scan top-5 peaks in human range (1-4 m → bins ~25-100)
    from scipy.signal import find_peaks
    scan_lo, scan_hi = 10, 60   # 0.75m to 4.5m at 7.5cm/bin
    region = avg_all_db[scan_lo:scan_hi]
    peak_idx, props = find_peaks(region, distance=8, height=np.percentile(region, 70))
    if len(peak_idx) == 0:
        peak_idx = [int(np.argmax(region))]
    # Sort by height, take top 5
    heights = [region[i] for i in peak_idx]
    top5 = sorted(zip(heights, peak_idx), reverse=True)[:5]

    print(f"  Top peaks in 1-4m range (sorted by strength):")
    print(f"  {'bin':>5s} {'range':>7s} {'dB':>6s}   {'TX0-RX phases':>40s}   {'ΔTX0-TX1':>10s}")

    for height, idx in top5:
        bin_idx = scan_lo + int(idx)
        range_m = bin_idx * 0.0751

        # Phases at this bin for TX0 (chirps[0::2]) and TX1 (chirps[1::2])
        tx0_phases = []
        tx1_phases = []
        for rx in range(NUM_RX):
            f0 = np.mean(np.fft.fft(tx0_chirps[:, rx, :], axis=-1), axis=0)[bin_idx]
            f1 = np.mean(np.fft.fft(tx2_chirps[:, rx, :], axis=-1), axis=0)[bin_idx]
            tx0_phases.append(float(np.degrees(np.angle(f0))))
            tx1_phases.append(float(np.degrees(np.angle(f1))))

        # Average phase difference TX0 → TX1 across RX channels
        tx_diff = np.mean([((tx1_phases[r] - tx0_phases[r] + 180) % 360 - 180)
                           for r in range(NUM_RX)])

        tx0_str = " ".join(f"{p:+5.0f}" for p in tx0_phases)
        print(f"  {bin_idx:>5d} {range_m:>6.2f}m {height:>5.1f}   [{tx0_str}]   {tx_diff:+7.1f}°")

    # Use strongest peak for detailed analysis
    peak_bin = scan_lo + top5[0][1]
    range_m = peak_bin * 0.0751
    print(f"\n  Detailed analysis at strongest peak: bin {peak_bin} (~{range_m:.2f} m)")
    print(f"  Virtual array phases (degrees):")

    phases = []
    for tx_idx, (tx_label, chirps) in enumerate([("TX0", tx0_chirps), ("TX1", tx2_chirps)]):
        for rx in range(NUM_RX):
            avg_fft_rx = np.mean(np.fft.fft(chirps[:, rx, :], axis=-1), axis=0)
            phase_deg = float(np.degrees(np.angle(avg_fft_rx[peak_bin])))
            virt_idx = tx_idx * 4 + rx
            phases.append(phase_deg)
            print(f"    v{virt_idx} ({tx_label}+RX{rx}): {phase_deg:+7.1f}°")

    # Check if phases are progressively shifting (indicating real spatial separation)
    phase_diffs = np.diff(phases)
    mean_diff = float(np.mean(phase_diffs))
    std_diff = float(np.std(phase_diffs))
    print(f"  Mean phase step: {mean_diff:.1f}° ± {std_diff:.1f}°")
    if std_diff < 5 and abs(mean_diff) > 1:
        print(f"  ✓ Progressive phase shift detected — virtual array is working")
    elif std_diff < 2 and abs(mean_diff) < 1:
        print(f"  ✗ All phases nearly identical — no spatial diversity")
        print(f"    → RX channels may be the same data duplicated")
    else:
        print(f"  ~ Phase pattern is irregular — check antenna connections")

    # ── Check 5b: RX phase ramp WITHIN each TX ──
    print(f"\n── Check 5b: RX-only phase ramp per TX ──")
    print(f"  For a target off-boresight, phase across RX should ramp linearly.")
    print(f"  Compare TX0-only ramp with TX2-only ramp.")

    for tx_label, chirps in [("TX0", tx0_chirps), ("TX2", tx2_chirps)]:
        rx_phases = []
        for rx in range(NUM_RX):
            avg_fft = np.mean(np.fft.fft(chirps[:, rx, :], axis=-1), axis=0)
            rx_phases.append(float(np.degrees(np.angle(avg_fft[peak_bin]))))
        # Unwrap for easier reading
        unwrapped = np.unwrap(np.radians(rx_phases))
        unwrapped_deg = np.degrees(unwrapped)
        steps = np.diff(unwrapped_deg)
        step_mean = float(np.mean(steps))
        step_std = float(np.std(steps))
        # For λ/2 spacing, phase step (deg) → angle θ from: step = 180 * sin(θ)
        est_angle = float(np.degrees(np.arcsin(np.clip(step_mean / 180.0, -1, 1))))
        phase_str = " ".join(f"{p:+6.1f}" for p in unwrapped_deg)
        print(f"  {tx_label}: [{phase_str}]  step={step_mean:+.1f}°±{step_std:.1f}° "
              f"→ θ≈{est_angle:+.1f}°")

    # ── Check 5c: Full 8-element angle FFT (what the fan display sees) ──
    print(f"\n── Check 5c: Angle spectrum from 8-element virtual array ──")
    virtual = np.zeros(8, dtype=np.complex64)
    for rx in range(NUM_RX):
        avg_tx0 = np.mean(np.fft.fft(tx0_chirps[:, rx, :], axis=-1), axis=0)[peak_bin]
        avg_tx2 = np.mean(np.fft.fft(tx2_chirps[:, rx, :], axis=-1), axis=0)[peak_bin]
        virtual[rx] = avg_tx0
        virtual[rx + 4] = avg_tx2

    # Angle FFT
    n_fft = 64
    spectrum = np.fft.fftshift(np.fft.fft(virtual, n=n_fft))
    mag_spec = np.abs(spectrum)
    # Convert bins to angles
    u = np.linspace(-1, 1, n_fft, endpoint=False)
    angles = np.degrees(np.arcsin(np.clip(u, -1, 1)))

    # Find peak and its -3dB width
    peak_ang_idx = int(np.argmax(mag_spec))
    peak_val = float(mag_spec[peak_ang_idx])
    peak_angle = float(angles[peak_ang_idx])
    # Half-power width
    half = peak_val / np.sqrt(2)
    left = peak_ang_idx
    while left > 0 and mag_spec[left] > half:
        left -= 1
    right = peak_ang_idx
    while right < len(mag_spec) - 1 and mag_spec[right] > half:
        right += 1
    hpbw = float(angles[right] - angles[left])
    # SNR of peak vs. background
    median_bg = float(np.median(mag_spec))
    snr_db = 20 * np.log10(peak_val / (median_bg + 1e-9))

    print(f"  Peak angle:  {peak_angle:+.1f}°")
    print(f"  Peak/median: {snr_db:.1f} dB")
    print(f"  HPBW:        {hpbw:.1f}°")
    # Sample the spectrum at a few angles
    print(f"  Angle spectrum sampled:")
    for target_deg in [-45, -30, -15, 0, 15, 30, 45]:
        ang_idx = int(np.argmin(np.abs(angles - target_deg)))
        val_db = 20 * np.log10(mag_spec[ang_idx] / (peak_val + 1e-9))
        bar = "█" * int(max(0, (val_db + 30) / 2))
        print(f"    {target_deg:+4d}°: {val_db:+6.1f} dB  {bar}")

    if snr_db < 6:
        print(f"  ⚠ Peak barely above noise — angle spectrum is spread out.")
        print(f"    This means the fan display will show lateral smearing.")
        print(f"    Likely causes:")
        print(f"    - TX0 and TX2 not actually producing distinct wavefronts")
        print(f"    - RX indexing is scrambled (wrong element positions)")
        print(f"    - Target has multiple angular components (multipath)")
    elif hpbw > 25:
        print(f"  ⚠ Peak is wide (>25°) — real spatial info present but noisy.")
    else:
        print(f"  ✓ Sharp angular peak — MIMO array is working correctly.")

    # ── Check 6: Frame-to-frame consistency ──
    if n_complete >= 2:
        print(f"\n── Check 6: Frame-to-frame consistency ──")
        frame2 = parse_frame(raw_data[RAW_BYTES_PER_FRAME:])
        for rx in range(NUM_RX):
            f1_rms = float(np.sqrt(np.mean(np.abs(frame[:, rx, :])**2)))
            f2_rms = float(np.sqrt(np.mean(np.abs(frame2[:, rx, :])**2)))
            ratio = f2_rms / (f1_rms + 1e-12)
            print(f"  RX{rx}: frame1 RMS={f1_rms:.1f}, frame2 RMS={f2_rms:.1f}, "
                  f"ratio={ratio:.3f} {'✓' if 0.5 < ratio < 2.0 else '✗ unstable'}")

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f" SUMMARY")
    print(f"{'═'*60}")

    # Check if data matches 1TX/1RX
    # If all RX identical and TX0==TX2, it's probably still 1TX/1RX
    all_rx_identical = all(
        float(np.abs(np.sum(fft_per_rx[0] * np.conj(fft_per_rx[i]))) /
              (np.sqrt(np.sum(np.abs(fft_per_rx[0])**2)) *
               np.sqrt(np.sum(np.abs(fft_per_rx[i])**2)))) > 0.99
        for i in range(1, NUM_RX)
    )

    tx_identical = float(np.abs(np.sum(tx0_chirps[0,0,:] * np.conj(tx2_chirps[0,0,:]))) /
                        (np.sqrt(np.sum(np.abs(tx0_chirps[0,0,:])**2)) *
                         np.sqrt(np.sum(np.abs(tx2_chirps[0,0,:])**2)) + 1e-12)) > 0.99

    if all_rx_identical:
        print(f"  ✗ All RX channels contain identical data")
        print(f"    Likely cause: only 1 RX is enabled in the channel config,")
        print(f"    or the DCA1000 is duplicating RX0 across all channels.")
        print(f"    → Check Channel Config: all 4 RX boxes must be checked")
    else:
        print(f"  ✓ RX channels are distinct — {NUM_RX} independent receivers")

    if tx_identical:
        print(f"  ✗ TX0 and TX2 chirps are identical")
        print(f"    Likely cause: both chirps fire the same TX, or only 1 TX is enabled.")
        print(f"    → In the main window Chirp section:")
        print(f"      Chirp 0: Set Start/End=0, check TX0 only, click Set")
        print(f"      Chirp 1: Set Start/End=1, check TX2 only, click Set")
        print(f"    → In Frame section: End Chirp TX = 1")
    else:
        print(f"  ✓ TX0 and TX2 chirps are distinct — TDM-MIMO is active")

    if not all_rx_identical and not tx_identical:
        print(f"\n  ✓✓ MIMO data looks correct — 2TX × 4RX = 8 virtual elements")
        print(f"     The angle-of-arrival processing should work.")
        print(f"     If the fan display still shows red lines, the issue")
        print(f"     may be in the background subtraction or the angle FFT code.")
    else:
        print(f"\n  The fan display shows red lines because there's no spatial")
        print(f"  diversity — the angle FFT sees the same data at every element,")
        print(f"  so it spreads energy uniformly across all angles.")


def main():
    ap = argparse.ArgumentParser(description="MIMO data diagnostic")
    ap.add_argument("--host", default=HOST_IP)
    ap.add_argument("--port", type=int, default=DATA_PORT)
    ap.add_argument("--file", type=str, default=None,
                    help="read from a saved binary file instead of UDP")
    ap.add_argument("--frames", type=int, default=NUM_FRAMES_TO_CAPTURE)
    ap.add_argument("--countdown", type=int, default=5,
                    help="seconds to wait before capturing (get in position)")
    args = ap.parse_args()

    if args.file:
        print(f"[FILE] reading from {args.file}")
        with open(args.file, "rb") as f:
            raw = f.read()
        print(f"[FILE] {len(raw):,} bytes")
    else:
        # Countdown so you can walk into position
        if args.countdown > 0:
            print("")
            print("═" * 55)
            print(f"  GET IN POSITION — capture starts in {args.countdown}s")
            print(f"  Stand ~2 m from radar, ~30° off boresight,")
            print(f"  clear of walls and other reflectors.")
            print("═" * 55)
            for i in range(args.countdown, 0, -1):
                print(f"  {i}...", flush=True)
                time.sleep(1)
            print(f"  GO — capturing now")
            print("")
        raw = capture_raw_data(args.host, args.port, args.frames)

    run_diagnostics(raw)


if __name__ == "__main__":
    main()