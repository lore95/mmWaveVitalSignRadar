"""
Parse adc_data_raw.bin captured with DCA1000 + currentconfigdca1000.xml.

Layout in the file (verified by inspection):
  - No per-packet headers; the file is concatenated UDP payloads.
  - Each ADC value is an int16 zero-padded to 32 bits (every odd int16 is 0).
    This is the DCA1000 2-lane capture format with only LVDS lane 1 active.
  - After dropping the zero-pad ints, the stream is I, Q, I, Q, ... per ADC sample.
  - Config: 1 RX, 1 TX, 256 ADC samples/chirp, 128 chirps/frame, complex 2x.
"""

import argparse
import numpy as np
from pathlib import Path

NUM_RX = 1
NUM_TX = 1
NUM_ADC_SAMPLES = 540
NUM_CHIRPS = 128  # loopCount in <apiname_frame_cfg>


def parse(in_path: str, out_path: str) -> np.ndarray:
    raw = np.fromfile("Bins/" + in_path, dtype=np.int16)

    # Drop the zero-pad int16s -> effective I/Q stream.
    samples = raw[0::2]
    pad = raw[1::2]
    # sanity: padding should be all zero
    nz = int(np.count_nonzero(pad))
    if nz:
        print(f"warning: {nz} pad slots are non-zero ({nz/len(pad):.4%}) - "
              "format assumption may be wrong")

    # Combine I/Q into complex samples.
    # samples layout: [I0, Q0, I1, Q1, ...]
    iq = samples[0::2].astype(np.float32) + 1j * samples[1::2].astype(np.float32)
    iq = iq.astype(np.complex64)

    per_frame = NUM_CHIRPS * NUM_TX * NUM_RX * NUM_ADC_SAMPLES
    n_frames = iq.size // per_frame
    leftover = iq.size - n_frames * per_frame
    print(f"effective complex samples: {iq.size}")
    print(f"frame size (complex samples): {per_frame}")
    print(f"complete frames: {n_frames}, leftover (dropped): {leftover}")

    iq = iq[: n_frames * per_frame].reshape(
        n_frames, NUM_CHIRPS * NUM_TX, NUM_RX, NUM_ADC_SAMPLES
    )
    print(f"output shape (frames, chirps*tx, rx, samples): {iq.shape}")
    out_path = "ParsedData/" + str(Path(args.in_raw_file).with_suffix(".npy"))

    np.save(out_path, iq)
    print(f"saved -> {out_path}")
    return iq


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--in_raw_file", default="adc_data_raw.bin")
    p.add_argument("-o", "--out_npy", default="data.npy")
    args = p.parse_args()
    
    parse(args.in_raw_file, args.out_npy)