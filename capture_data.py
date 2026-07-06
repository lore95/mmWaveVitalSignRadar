"""
record_session.py
==================
Records raw radar IQ data + synchronized metadata for later offline
analysis (e.g. looking for person-identifying patterns in gait, breathing,
or micro-Doppler signatures).
 
Built on top of the same UDPReceiver/format-check logic as radar_vitals.py
(matching the CORRECTED 3TX/4RX Lua config: 256 ADC samples, 64 chirp loops,
10000 ksps, all 4 LVDS lanes). If you change the Lua file again, update the
CONFIG dict below to match - the metadata sidecar is what lets future-you
(or a teammate) parse old recordings correctly even after the live config
has moved on.
 
Why a metadata sidecar matters here specifically: pattern-analysis /
person-ID work is exactly the case where a silent parsing mismatch (wrong
NUM_RX, wrong sample rate) won't crash anything - it'll just quietly bake
a wrong assumption into "signatures" that look plausible but are actually
artifacts of misparsing. Every recording gets a frozen-in-time JSON of the
config it was captured under, plus a per-frame timestamp/seq log so frame
drops are visible in the data instead of silently shifting time alignment.
 
Usage:
  sudo python record_session.py --subject alex --label walking_in_front --duration 60
  sudo python record_session.py --subject alex --label sitting_breathing --duration 120 --notes "2m, facing radar"
 
Output (per session, under SessionData/<timestamp>_<subject>_<label>/):
  raw_adc.bin       - concatenated raw ADC payload bytes (same format as
                       listenonport.py's output - parse with
                       radar_vitals.py's `parse` subcommand)
  meta.json         - capture config, subject/label/notes, start/end time
  frame_log.jsonl   - one line per UDP packet: seq number, wall-clock time,
                       running missed-packet count (lets you re-align /
                       discard frames around any dropouts during analysis)
"""
 
import argparse
import json
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
 
# ── must match the CORRECTED Lua config (see 1843_config_streaming_task4.lua) ──
CONFIG = {
    "host_ip": "192.168.33.30",
    "data_port": 4098,
    "num_tx": 3,
    "num_rx": 4,
    "num_adc_samples": 256,
    "chirp_loops": 64,
    "sample_rate_ksps": 10000,
    "freq_slope_mhz_us": 78.020,
    "f0_hz": 77.0e9,
    "periodicity_ms": 20,
    "chirp_tx_order": [1, 0, 2],   # transmission order -> physical TX index
    "lane_layout": "sample_major_lane_interleaved",  # [RX0_I,RX0_Q,RX1_I,RX1_Q,...] per sample
    "header_size_bytes": 10,
}
 
HEADER_SIZE = CONFIG["header_size_bytes"]
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True,
                     help="identifier for the person being recorded (for your own labeling, "
                          "not transmitted anywhere)")
    ap.add_argument("--label", required=True,
                     help="short description of the activity/scenario, e.g. 'walking', "
                          "'sitting_still', 'standing_breathing'")
    ap.add_argument("--notes", default="",
                     help="free-text notes: distance, orientation, clothing, anything that "
                          "might matter for pattern analysis later")
    ap.add_argument("--duration", type=int, default=0,
                     help="seconds to record, 0 = until Ctrl+C")
    ap.add_argument("--out-root", default="SessionData")
    ap.add_argument("--host", default=CONFIG["host_ip"])
    ap.add_argument("--port", type=int, default=CONFIG["data_port"])
    args = ap.parse_args()
 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.out_root) / f"{ts}_{args.subject}_{args.label}"
    session_dir.mkdir(parents=True, exist_ok=True)
 
    bin_path = session_dir / "raw_adc.bin"
    meta_path = session_dir / "meta.json"
    frame_log_path = session_dir / "frame_log.jsonl"
 
    print(f"[record] session dir: {session_dir}")
    print(f"[record] subject={args.subject!r} label={args.label!r}")
    if args.notes:
        print(f"[record] notes: {args.notes}")
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.settimeout(5.0)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f"ERROR: cannot bind {args.host}:{args.port} - {e}")
        print(f"Try: sudo python {sys.argv[0]} ...")
        sys.exit(1)
 
    print(f"[record] listening on {args.host}:{args.port}")
    if args.duration > 0:
        print(f"[record] will stop after {args.duration}s")
    print("[record] press Ctrl+C to stop early\n")
 
    outfile = open(bin_path, "wb")
    framelog = open(frame_log_path, "w")
 
    packets = 0
    total_bytes = 0
    last_seq = None
    missed_total = 0
    start_wall = time.time()
    last_print = start_wall
    running = True
 
    import signal
 
    def stop(_sig, _frame):
        nonlocal running
        running = False
 
    signal.signal(signal.SIGINT, stop)
 
    while running:
        elapsed = time.time() - start_wall
        if args.duration > 0 and elapsed >= args.duration:
            break
 
        try:
            data, _addr = sock.recvfrom(2048)
        except socket.timeout:
            print("  waiting for packets...")
            continue
        except OSError:
            break
 
        wall_now = time.time()
        packets += 1
        total_bytes += len(data)
 
        if len(data) <= HEADER_SIZE:
            continue
 
        seq = struct.unpack("<I", data[0:4])[0]
        if last_seq is not None and seq != last_seq + 1:
            missed_total += max(0, seq - last_seq - 1)
        last_seq = seq
 
        payload = data[HEADER_SIZE:]
        outfile.write(payload)
 
        framelog.write(json.dumps({
            "packet_idx": packets,
            "seq": seq,
            "wall_time": wall_now,
            "missed_total": missed_total,
            "payload_bytes": len(payload),
        }) + "\n")
 
        if wall_now - last_print >= 1.0:
            rate_mbps = (total_bytes * 8) / (elapsed * 1e6) if elapsed > 0 else 0
            print(f"  [{elapsed:6.1f}s] packets={packets:>8d}  "
                  f"bytes={total_bytes/1e6:>8.2f}MB  rate={rate_mbps:>5.1f}Mbps  "
                  f"missed={missed_total}", flush=True)
            last_print = wall_now
 
    end_wall = time.time()
    outfile.close()
    framelog.close()
    sock.close()
 
    meta = {
        "subject": args.subject,
        "label": args.label,
        "notes": args.notes,
        "start_wall_time": start_wall,
        "end_wall_time": end_wall,
        "duration_s": end_wall - start_wall,
        "packets": packets,
        "total_bytes": total_bytes,
        "missed_packets": missed_total,
        "config": CONFIG,
        "files": {
            "raw_adc": bin_path.name,
            "frame_log": frame_log_path.name,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
 
    print(f"\n{'='*60}")
    print(f"  session complete")
    print(f"  duration:  {meta['duration_s']:.1f}s")
    print(f"  packets:   {packets}  (missed: {missed_total})")
    print(f"  raw data:  {total_bytes/1e6:.2f} MB -> {bin_path}")
    print(f"  metadata:  {meta_path}")
    print(f"{'='*60}")
    print(f"\nTo parse later:")
    print(f"  python radar_vitals.py parse --in {bin_path} --out {session_dir/'parsed.npy'}")
 
 
if __name__ == "__main__":
    main()