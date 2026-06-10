#!/usr/bin/env python3
"""
Connect to a Vernier Go Direct Respiration Belt (GDX-RB) via BLE
and print every sensor reading to the terminal.

This is a discovery script — it enables ALL channels so you can see
exactly what the device sends, the data types, units, and update rates.

Usage:
    pip install godirect
    python gdx_rb_dump.py              # auto-find nearest device
    python gdx_rb_dump.py --list       # list all nearby Go Direct devices
    python gdx_rb_dump.py --usb        # connect via USB instead of BLE

Press Ctrl+C to stop.
"""

import argparse
import time
import sys
import logging


def main():
    parser = argparse.ArgumentParser(description="GDX-RB raw data dump")
    parser.add_argument("--list", action="store_true",
                        help="scan and list all nearby Go Direct BLE devices, then exit")
    parser.add_argument("--usb", action="store_true",
                        help="connect via USB instead of BLE")
    parser.add_argument("--period", type=int, default=100,
                        help="collection period in ms (default: 100 = 10 Hz)")
    parser.add_argument("--all-channels", action="store_true",
                        help="enable ALL channels (Force, Resp Rate, Steps, Step Rate). "
                             "Default enables only the device defaults (Force + Resp Rate).")
    parser.add_argument("--debug", action="store_true",
                        help="enable godirect debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig()
        logging.getLogger("godirect").setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Import godirect (fail early with a clear message)
    # ------------------------------------------------------------------
    try:
        from godirect import GoDirect
    except ImportError:
        print("ERROR: godirect module not found.")
        print("  Install it with:  pip install godirect")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Initialise — choose USB, BLE, or both
    # ------------------------------------------------------------------
    if args.usb:
        godirect = GoDirect(use_ble=False, use_usb=True)
        mode = "USB"
    else:
        godirect = GoDirect(use_ble=True, use_usb=False)
        mode = "BLE"

    print(f"[init] godirect ready, mode={mode}")

    # ------------------------------------------------------------------
    # List / scan
    # ------------------------------------------------------------------
    if args.list:
        print(f"\n[scan] searching for Go Direct devices over {mode}…")
        devices = godirect.list_devices()
        if not devices:
            print("  No devices found. Is the belt powered on?")
        else:
            for i, d in enumerate(devices):
                print(f"  [{i}] {d.name}  (order code: {d.order_code})")
        godirect.quit()
        return

    # ------------------------------------------------------------------
    # Auto-find the nearest device
    # ------------------------------------------------------------------
    print(f"\n[scan] looking for the nearest Go Direct device over {mode}…")
    print("       (make sure the belt LED is flashing red)")
    device = godirect.get_device()

    if device is None:
        print("ERROR: no device found. Check that the belt is powered on and in range.")
        godirect.quit()
        sys.exit(1)

    print(f"\n[found] {device.name}")
    print(f"        order code : {device.order_code}")
    print(f"        serial     : {device.serial_number}")

    # ------------------------------------------------------------------
    # Open and discover all available sensors
    # ------------------------------------------------------------------
    if not device.open():
        print("ERROR: could not open device.")
        godirect.quit()
        sys.exit(1)

    # Print every sensor the device exposes (enabled or not)
    print("\n── available sensor channels ──")
    all_sensors = device.list_sensors()
    for s_info in all_sensors:
        # s_info is typically a dict or sensor-info object;
        # the exact structure varies by godirect version, so we just print it
        print(f"  {s_info}")

    # Enable channels
    if args.all_channels:
        # Enable every channel the device has
        sensor_numbers = [s["number"] if isinstance(s, dict) else s
                          for s in all_sensors]
        print(f"\n[config] enabling ALL channels: {sensor_numbers}")
        device.enable_sensors(sensor_numbers)
    else:
        print("\n[config] using default enabled channels (Force + Respiration Rate)")
        device.enable_default_sensors()

    enabled = device.get_enabled_sensors()
    print(f"[config] enabled sensors ({len(enabled)}):")
    for s in enabled:
        print(f"  ch {s.sensor_number:>2d}  {s.sensor_description:<25s}  [{s.sensor_units}]")

    # ------------------------------------------------------------------
    # Start collection and dump readings
    # ------------------------------------------------------------------
    print(f"\n[start] collecting at {args.period} ms period ({1000/args.period:.1f} Hz)")
    print("        press Ctrl+C to stop\n")

    device.start(period=args.period)

    # Header
    header_parts = ["sample", "wall_time"]
    for s in enabled:
        header_parts.append(f"{s.sensor_description} ({s.sensor_units})")
    header = "  |  ".join(header_parts)
    print(header)
    print("─" * len(header))

    sample_num = 0
    t0 = time.monotonic()

    try:
        while True:
            if device.read():
                sample_num += 1
                wall = time.monotonic() - t0
                parts = [f"{sample_num:>6d}", f"{wall:>10.3f}"]
                for s in enabled:
                    val = s.value
                    if val is None or (isinstance(val, float) and val != val):
                        # NaN check (val != val is the classic float NaN test)
                        parts.append(f"{'NaN':>12s}")
                    else:
                        parts.append(f"{val:>12.4f}")
                print("  |  ".join(parts))
    except KeyboardInterrupt:
        print(f"\n\n[stop] {sample_num} samples in {time.monotonic()-t0:.1f} s")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    device.stop()
    device.close()
    godirect.quit()
    print("[done]")


if __name__ == "__main__":
    main()