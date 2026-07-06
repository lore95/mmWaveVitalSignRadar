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
NUM_TX           = 2
NUM_CHIRPS       = 64      # chirp loops
NUM_ADC_SAMPLES  = 540
TOTAL_CHIRPS     = NUM_CHIRPS * NUM_TX  # 128

# Each complex sample = I(16) + pad(16) + Q(16) + pad(16) = 8 bytes
COMPLEX_PER_FRAME = TOTAL_CHIRPS * NUM_RX * NUM_ADC_SAMPLES  # 276,480
RAW_BYTES_PER_FRAME = COMPLEX_PER_FRAME * 8                   # 2,211,840

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
    """Parse one frame into (TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES) complex."""
    raw = np.frombuffer(raw_bytes[:RAW_BYTES_PER_FRAME], dtype=np.int16)
    samples = raw[0::2]  # drop zero-pad
    iq = samples[0::2].astype(np.float32) + 1j * samples[1::2].astype(np.float32)
    return iq.reshape(TOTAL_CHIRPS, NUM_RX, NUM_ADC_SAMPLES)


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
    tx0_chirps = frame[0::2]  # even indices
    tx2_chirps = frame[1::2]  # odd indices
    print(f"  TX0 chirps shape: {tx0_chirps.shape}  (expected ({NUM_CHIRPS}, {NUM_RX}, {NUM_ADC_SAMPLES}))")
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
            peak_bin = int(np.argmax(mag[5:]))  + 5  # skip DC
            peak_val = float(mag[peak_bin])
            print(f"  bin{peak_bin:3d}={peak_val:5.0f}", end="")
        print()

    # ── Check 5: Phase across virtual array ──
    print(f"\n── Check 5: Phase across virtual array at peak ──")
    print(f"  (should show progressive phase shift if antenna separation is real)")

    # Find the overall peak range bin
    all_fft = np.fft.fft(frame[:, :, :], axis=-1)[..., :NUM_ADC_SAMPLES // 2]
    avg_all = np.abs(all_fft.mean(axis=(0, 1)))
    peak_bin = int(np.argmax(avg_all[5:])) + 5

    print(f"  Peak range bin: {peak_bin}")
    print(f"  Virtual array phases (degrees):")

    phases = []
    for tx_idx, (tx_label, chirps) in enumerate([("TX0", tx0_chirps), ("TX2", tx2_chirps)]):
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
    args = ap.parse_args()

    if args.file:
        print(f"[FILE] reading from {args.file}")
        with open(args.file, "rb") as f:
            raw = f.read()
        print(f"[FILE] {len(raw):,} bytes")
    else:
        raw = capture_raw_data(args.host, args.port, args.frames)

    run_diagnostics(raw)


if __name__ == "__main__":
    main()