#!/usr/bin/env python3
"""
Minimal DCA1000 UDP listener — captures raw ADC packets and saves to file.
No DCA1000 commands, no serial, no processing. Just listens.

Usage:
  sudo python udp_capture.py
  sudo python udp_capture.py --output my_capture.bin
  sudo python udp_capture.py --duration 30
"""

import socket
import struct
import time
import sys
import argparse
import signal

HOST_IP   = "192.168.33.30"
DATA_PORT = 4098

def main():
    parser = argparse.ArgumentParser(description="DCA1000 UDP raw data capture")
    parser.add_argument('--output', type=str, default='adc_data_raw.bin',
                        help='Output file (default: adc_data_raw.bin)')
    parser.add_argument('--duration', type=int, default=0,
                        help='Capture duration in seconds (0 = until Ctrl+C)')
    parser.add_argument('--host', type=str, default=HOST_IP,
                        help=f'Host IP (default: {HOST_IP})')
    parser.add_argument('--port', type=int, default=DATA_PORT,
                        help=f'UDP port (default: {DATA_PORT})')
    args = parser.parse_args()

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)  # 4MB buffer
    sock.settimeout(5.0)

    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f"ERROR: Cannot bind to {args.host}:{args.port}")
        print(f"  {e}")
        print(f"\nTry: sudo python {sys.argv[0]}")
        sys.exit(1)

    print(f"Listening on {args.host}:{args.port}")
    print(f"Saving to: {args.output}")
    if args.duration > 0:
        print(f"Duration: {args.duration} seconds")
    print(f"Press Ctrl+C to stop\n")

    outfile = open("Bins/" + args.output, 'wb')
    packets = 0
    total_bytes = 0
    adc_bytes = 0
    start_time = time.time()
    last_print = start_time

    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)

    while running:
        # Check duration
        elapsed = time.time() - start_time
        if args.duration > 0 and elapsed >= args.duration:
            break

        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            print("  Waiting for packets...")
            continue
        except OSError:
            break

        packets += 1
        total_bytes += len(data)

        # DCA1000 packet: 10-byte header + ADC payload
        # Header: 4 bytes seq_num (uint32 LE) + 6 bytes byte_count
        if len(data) > 10:
            seq_num = struct.unpack('<I', data[0:4])[0]
            payload = data[10:]
            adc_bytes += len(payload)
            outfile.write(payload)

        # Print stats every second
        now = time.time()
        if now - last_print >= 1.0:
            elapsed = now - start_time
            rate_mbps = (total_bytes * 8) / (elapsed * 1e6) if elapsed > 0 else 0
            print(f"  [{elapsed:6.1f}s] Packets: {packets:>8d}  |  "
                  f"ADC data: {adc_bytes / 1e6:>8.2f} MB  |  "
                  f"Rate: {rate_mbps:>5.1f} Mbps  |  "
                  f"Last seq: {seq_num}", flush=True)
            last_print = now

    # Done
    elapsed = time.time() - start_time
    outfile.close()
    sock.close()

    print(f"\n{'='*60}")
    print(f"  Capture complete")
    print(f"  Duration:    {elapsed:.1f} seconds")
    print(f"  Packets:     {packets}")
    print(f"  ADC data:    {adc_bytes / 1e6:.2f} MB")
    print(f"  Saved to:    {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()