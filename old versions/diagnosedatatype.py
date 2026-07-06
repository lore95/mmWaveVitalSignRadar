#!/usr/bin/env python3
"""
DCA1000 Data Format Diagnostic
================================
Captures a few frames, tries all three possible I/Q data formats,
and prints analysis to help determine which format is correct.

Run while DCA1000 is streaming, then paste the output to Claude.

Usage:
  sudo python diagnose_format.py
"""

import socket
import struct
import numpy as np
import sys
import time

HOST_IP   = "192.168.33.30"
DATA_PORT = 4098

NUM_ADC_SAMPLES      = 256
NUM_CHIRPS_PER_FRAME = 128
NUM_RX               = 1
FREQ_SLOPE_MHZ_US    = 29.982
SAMPLE_RATE_KSPS     = 10000

BYTES_PER_FRAME = NUM_ADC_SAMPLES * 4 * NUM_RX * NUM_CHIRPS_PER_FRAME
CHIRP_DURATION_US = NUM_ADC_SAMPLES / (SAMPLE_RATE_KSPS / 1000.0)
BANDWIDTH_MHZ = FREQ_SLOPE_MHZ_US * CHIRP_DURATION_US
RANGE_RES_M = 3e8 / (2 * BANDWIDTH_MHZ * 1e6)

HEADER_SIZE = 10
NUM_FRAMES_TO_CAPTURE = 5


def capture_frames():
    """Capture raw frames from DCA1000."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.settimeout(5.0)
    try:
        sock.bind((HOST_IP, DATA_PORT))
    except OSError as e:
        print(f"ERROR: Cannot bind to {HOST_IP}:{DATA_PORT} — {e}")
        print(f"Try: sudo python {sys.argv[0]}")
        sys.exit(1)

    print(f"Listening on {HOST_IP}:{DATA_PORT}...")
    print(f"Capturing {NUM_FRAMES_TO_CAPTURE} frames...\n")

    buf = bytearray()
    frames = []
    packets = 0
    first_seq = None
    seq_nums = []

    while len(frames) < NUM_FRAMES_TO_CAPTURE:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            print("  Timeout — no data. Is the radar streaming?")
            sock.close()
            sys.exit(1)

        packets += 1
        if len(data) > HEADER_SIZE:
            seq = struct.unpack('<I', data[0:4])[0]
            byte_count = struct.unpack('<Q', data[4:10] + b'\x00\x00')[0]
            if first_seq is None:
                first_seq = seq
            seq_nums.append(seq)
            payload = data[HEADER_SIZE:]
            buf.extend(payload)

        while len(buf) >= BYTES_PER_FRAME:
            frames.append(bytes(buf[:BYTES_PER_FRAME]))
            buf = buf[BYTES_PER_FRAME:]

    sock.close()
    return frames, packets, seq_nums


def parse_format_interleaved(raw):
    """I0, Q0, I1, Q1, ... per chirp"""
    data = raw.reshape(NUM_CHIRPS_PER_FRAME, NUM_ADC_SAMPLES * 2 * NUM_RX)
    i = data[:, 0::2].astype(np.float64)
    q = data[:, 1::2].astype(np.float64)
    return i + 1j * q


def parse_format_ti_reorder(raw):
    """TI LVDS reorder: [I0, I1, Q0, Q1] groups → [I0, Q0, I1, Q1]"""
    r = raw.copy().reshape(-1, 4)
    reordered = np.zeros_like(r)
    reordered[:, 0] = r[:, 0]  # I0
    reordered[:, 1] = r[:, 2]  # Q0
    reordered[:, 2] = r[:, 1]  # I1
    reordered[:, 3] = r[:, 3]  # Q1
    flat = reordered.reshape(-1)
    flat = flat.reshape(NUM_CHIRPS_PER_FRAME, NUM_ADC_SAMPLES * 2 * NUM_RX)
    i = flat[:, 0::2].astype(np.float64)
    q = flat[:, 1::2].astype(np.float64)
    return i + 1j * q


def parse_format_iq_split(raw):
    """All I first, then all Q per chirp"""
    data = raw.reshape(NUM_CHIRPS_PER_FRAME, NUM_ADC_SAMPLES * 2 * NUM_RX)
    i = data[:, :NUM_ADC_SAMPLES].astype(np.float64)
    q = data[:, NUM_ADC_SAMPLES:].astype(np.float64)
    return i + 1j * q


def analyze_format(name, complex_data, frame_idx):
    """Analyze a parsed frame and return metrics."""
    window = np.hanning(NUM_ADC_SAMPLES)
    windowed = complex_data * window[np.newaxis, :]
    range_fft = np.fft.fft(windowed, axis=1)
    range_fft = range_fft[:, :NUM_ADC_SAMPLES // 2]

    range_profile = np.mean(np.abs(range_fft), axis=0)

    # Skip bin 0 (DC)
    peak_bin = 1 + np.argmax(range_profile[1:])
    peak_val = range_profile[peak_bin]
    peak_range_m = peak_bin * RANGE_RES_M

    # Noise floor (exclude peak region)
    noise_mask = np.ones(len(range_profile), dtype=bool)
    noise_mask[0] = False
    noise_mask[max(0, peak_bin - 3):min(len(range_profile), peak_bin + 4)] = False
    noise_floor = np.mean(range_profile[noise_mask])
    snr = 20 * np.log10(peak_val / noise_floor) if noise_floor > 0 else 0

    # Phase stability across chirps at peak bin
    phases = np.angle(range_fft[:, peak_bin])
    phases_unwrapped = np.unwrap(phases)
    phase_std = np.std(phases_unwrapped)

    # Range profile sharpness: ratio of peak to second-highest peak outside ±3 bins
    profile_copy = range_profile.copy()
    profile_copy[max(0, peak_bin - 3):min(len(profile_copy), peak_bin + 4)] = 0
    profile_copy[0] = 0
    second_peak = np.max(profile_copy)
    sharpness = peak_val / second_peak if second_peak > 0 else 999

    return {
        'peak_bin': peak_bin,
        'peak_range_m': peak_range_m,
        'peak_val': peak_val,
        'snr_db': snr,
        'phase_std': phase_std,
        'sharpness': sharpness,
        'range_profile': range_profile,
    }


def print_range_profile(profile, peak_bin, max_bins=30):
    """Print ASCII range profile."""
    mx = np.max(profile[1:]) if np.max(profile[1:]) > 0 else 1
    end = min(max_bins, len(profile))
    for i in range(1, end):
        r = i * RANGE_RES_M
        bar_len = int(40 * profile[i] / mx)
        bar = '█' * bar_len
        marker = " ◄ PEAK" if i == peak_bin else ""
        print(f"    {r:5.2f}m │{bar}{marker}")


def main():
    print("=" * 64)
    print("  DCA1000 Data Format Diagnostic")
    print("=" * 64)
    print(f"  Config: {NUM_ADC_SAMPLES} samples, {NUM_CHIRPS_PER_FRAME} chirps, "
          f"{NUM_RX} RX, {SAMPLE_RATE_KSPS} ksps")
    print(f"  Range resolution: {RANGE_RES_M:.4f} m")
    print(f"  Expected frame size: {BYTES_PER_FRAME} bytes")
    print()

    # Capture
    frames, total_packets, seq_nums = capture_frames()
    print(f"  Captured {len(frames)} frames from {total_packets} packets")

    # Packet header analysis
    print(f"\n  --- Packet Header Analysis ---")
    print(f"  First seq#: {seq_nums[0]}, Last seq#: {seq_nums[-1]}")
    diffs = np.diff(seq_nums)
    print(f"  Seq# diffs — min: {np.min(diffs)}, max: {np.max(diffs)}, "
          f"mean: {np.mean(diffs):.1f}")
    missed = np.sum(diffs > 1)
    print(f"  Missed packets: {missed} ({100*missed/len(diffs):.1f}%)")

    # Raw data analysis
    print(f"\n  --- Raw Data Analysis (Frame 0) ---")
    raw0 = np.frombuffer(frames[0], dtype=np.int16)
    print(f"  Total int16 values: {len(raw0)}")
    print(f"  Min: {np.min(raw0)}, Max: {np.max(raw0)}, Mean: {np.mean(raw0):.1f}")
    print(f"  Std: {np.std(raw0):.1f}")
    print(f"  First 20 values: {raw0[:20].tolist()}")
    print(f"  Values at [0,1,2,3]: {raw0[0]}, {raw0[1]}, {raw0[2]}, {raw0[3]}")
    print(f"  Values at [4,5,6,7]: {raw0[4]}, {raw0[5]}, {raw0[6]}, {raw0[7]}")

    # Check for patterns
    print(f"\n  --- Zero/Pattern Check ---")
    zeros = np.sum(raw0 == 0)
    print(f"  Zero values: {zeros} / {len(raw0)} ({100*zeros/len(raw0):.1f}%)")

    # Check if alternating values look like I/Q or like I,I,Q,Q
    even = raw0[0::2]  # positions 0,2,4,...
    odd  = raw0[1::2]  # positions 1,3,5,...
    print(f"  Even positions — mean: {np.mean(even):.1f}, std: {np.std(even):.1f}")
    print(f"  Odd positions  — mean: {np.mean(odd):.1f}, std: {np.std(odd):.1f}")

    grp4_0 = raw0[0::4]  # positions 0,4,8,...
    grp4_1 = raw0[1::4]  # positions 1,5,9,...
    grp4_2 = raw0[2::4]  # positions 2,6,10,...
    grp4_3 = raw0[3::4]  # positions 3,7,11,...
    print(f"  Group4 pos0 — mean: {np.mean(grp4_0):.1f}, std: {np.std(grp4_0):.1f}")
    print(f"  Group4 pos1 — mean: {np.mean(grp4_1):.1f}, std: {np.std(grp4_1):.1f}")
    print(f"  Group4 pos2 — mean: {np.mean(grp4_2):.1f}, std: {np.std(grp4_2):.1f}")
    print(f"  Group4 pos3 — mean: {np.mean(grp4_3):.1f}, std: {np.std(grp4_3):.1f}")

    # Try all three formats
    formats = {
        'interleaved': parse_format_interleaved,
        'ti_reorder':  parse_format_ti_reorder,
        'iq_split':    parse_format_iq_split,
    }

    print(f"\n{'='*64}")
    print(f"  FORMAT COMPARISON (using frame 0)")
    print(f"{'='*64}")

    results = {}
    for name, parser in formats.items():
        raw = np.frombuffer(frames[0], dtype=np.int16)
        complex_data = parser(raw)
        r = analyze_format(name, complex_data, 0)
        results[name] = r

        print(f"\n  --- Format: {name} ---")
        print(f"  Peak bin: {r['peak_bin']}  →  Range: {r['peak_range_m']:.2f} m")
        print(f"  SNR: {r['snr_db']:.1f} dB")
        print(f"  Peak/2nd ratio: {r['sharpness']:.2f}")
        print(f"  Phase std across chirps: {r['phase_std']:.4f} rad")
        print(f"  Range profile:")
        print_range_profile(r['range_profile'], r['peak_bin'])

    # Cross-frame consistency check
    print(f"\n{'='*64}")
    print(f"  CROSS-FRAME CONSISTENCY (peak bin across {len(frames)} frames)")
    print(f"{'='*64}")

    for name, parser in formats.items():
        bins = []
        ranges = []
        snrs = []
        for f in frames:
            raw = np.frombuffer(f, dtype=np.int16)
            c = parser(raw)
            r = analyze_format(name, c, 0)
            bins.append(r['peak_bin'])
            ranges.append(r['peak_range_m'])
            snrs.append(r['snr_db'])

        print(f"\n  {name}:")
        print(f"    Peak bins: {bins}")
        print(f"    Ranges:    {[f'{r:.2f}' for r in ranges]}")
        print(f"    SNRs:      {[f'{s:.1f}' for s in snrs]}")
        print(f"    Bin changes: {sum(1 for i in range(1,len(bins)) if bins[i]!=bins[i-1])}")

    # Recommendation
    print(f"\n{'='*64}")
    print(f"  RECOMMENDATION")
    print(f"{'='*64}")

    best = max(results.items(), key=lambda x: x[1]['snr_db'] + x[1]['sharpness'])
    print(f"\n  Best format: {best[0]}")
    print(f"    SNR: {best[1]['snr_db']:.1f} dB, Sharpness: {best[1]['sharpness']:.2f}")
    print(f"    Peak range: {best[1]['peak_range_m']:.2f} m")
    print(f"\n  Does {best[1]['peak_range_m']:.2f} m match your actual distance to the radar?")
    print(f"  If not, try the other formats and see which distance is correct.")
    print()


if __name__ == '__main__':
    main()