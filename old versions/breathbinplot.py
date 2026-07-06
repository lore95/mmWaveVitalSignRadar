"""
Two-panel synchronized animation:
  top:    range profile (post-bg, dB) for the current frame
  bottom: chest-displacement waveform (mm) over the whole recording, with a
          moving cursor at the current frame; title shows breathing-rate estimate
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import butter, filtfilt
from pathlib import Path

# ---- radar params ----

C = 2.998e8

SLOPE_HZ_PER_S = 70.006e12

SAMPLE_RATE_HZ = 10e6

F0_HZ = 77.0000000238419e9

LAMBDA_M = C / F0_HZ

NUM_ADC = 540

FRAME_PERIOD_S = 39.99975e-3

FPS_HZ = 1.0 / FRAME_PERIOD_S   # ~25 Hz

BG_FRAMES = 50

GIF_FPS = 25

XMAX_M = 2.0

def range_axis(n: int) -> np.ndarray:
    return np.arange(n // 2) * (SAMPLE_RATE_HZ / n) * C / (2 * SLOPE_HZ_PER_S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--in_npy", default="data.npy")
    p.add_argument("-o", "--out_gif", default="breathing.gif")
    p.add_argument("--no-chirp-avg", action="store_true",
                   help="use only chirp 0 of each frame instead of "
                        "coherent-averaging all 128 chirps")
    p.add_argument("--no-window", action="store_true",
                   help="skip the Hann window + DC removal before the range FFT")
    args = p.parse_args()

    cube = np.load("ParsedData/" + args.in_npy)         # (frames, chirps, rx, samples)
    n_frames, n_chirps, n_rx, n_samp = cube.shape
    print(f"cube: {cube.shape}, duration = {n_frames*FRAME_PERIOD_S:.1f} s")

    # ---- range FFT ----
    if args.no_window:
        print("range FFT: no Hann window, no DC removal")
        rfft = np.fft.fft(cube, axis=-1)[..., : n_samp // 2]
    else:
        print("range FFT: Hann window + DC removal")
        win = np.hanning(n_samp).astype(np.float32)
        cube_dc = cube - cube.mean(axis=-1, keepdims=True)
        rfft = np.fft.fft(cube_dc * win, axis=-1)[..., : n_samp // 2]

    rng = range_axis(n_samp)
    # auto-pick the dominant range bin near 0.5 m so the marker actually
    # sits on the peak, not on the rounded-off 0.5 m label.
    search_lo = max(1, int(np.searchsorted(rng, 0.2)))   # skip DC
    search_hi = int(np.searchsorted(rng, 1.5))
    avg_mag = np.abs(rfft).mean(axis=(0, 1)).squeeze()   # avg over frames+chirps
    target_bin = search_lo + int(np.argmax(avg_mag[search_lo:search_hi]))
    target_range_m = float(rng[target_bin])
    print(f"target bin {target_bin} -> {target_range_m:.3f} m  (auto, peak in 0.2-1.5 m)")

    # ---- range-profile magnitude (top panel), bg-subtracted ----
    bg = rfft[:BG_FRAMES].mean(axis=(0, 1), keepdims=True)
    mag = np.abs(rfft - bg).mean(axis=1).squeeze(1)
    mag_db = 20 * np.log10(mag + 1e-6)

    # ---- phase -> displacement at the target bin (bottom panel) ----
    # Use the raw (not bg-subtracted) complex - we want the absolute phase.
    if args.no_chirp_avg:
        print("phase from chirp 0 only (no chirp averaging)")
        z = rfft[:, 0, 0, target_bin]                 # (n_frames,) complex
    else:
        print("phase from coherent mean across all 128 chirps")
        z = rfft[:, :, 0, target_bin].mean(axis=1)    # (n_frames,) complex
    phase = np.unwrap(np.angle(z))

    t = np.arange(n_frames) * FRAME_PERIOD_S
    # remove linear drift (slow body motion / temperature)
    a, b = np.polyfit(t, phase, 1)
    phase_dt = phase - (a * t + b)
    # phase -> chest displacement (mm).  d = -lambda * phi / (4*pi)
    disp_mm = -LAMBDA_M * phase_dt / (4 * np.pi) * 1000.0

    # bandpass 0.1-0.6 Hz to isolate breathing
    bp_b, bp_a = butter(4, [0.1, 0.6], btype="bandpass", fs=FPS_HZ)
    disp_bp = filtfilt(bp_b, bp_a, disp_mm)

    # ---- breathing-rate estimate (post-bg region) ----
    sig = disp_bp[BG_FRAMES:] - disp_bp[BG_FRAMES:].mean()
    f_axis = np.fft.rfftfreq(len(sig), d=1 / FPS_HZ)
    spec = np.abs(np.fft.rfft(sig))
    band = (f_axis >= 0.1) & (f_axis <= 0.6)
    peak_f = float(f_axis[band][np.argmax(spec[band])])
    bpm = peak_f * 60
    print(f"breathing rate = {peak_f:.3f} Hz  ->  {bpm:.1f} BPM")

    # ---- restrict top panel to 0-2 m ----
    keep = rng <= XMAX_M
    rng2 = rng[keep]
    test_mag = mag_db[BG_FRAMES:][:, keep]            # post-bg only

    # ---- animation ----
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.4))

    # top
    line_r, = axs[0].plot(rng2, test_mag[0], lw=1.4)
    axs[0].axvline(target_range_m, color="r", ls="--", lw=0.8,
                   label=f"peak bin @ {target_range_m:.2f} m")
    axs[0].set_xlim(0, XMAX_M)
    axs[0].set_ylim(float(test_mag.min()) - 2, float(test_mag.max()) + 2)
    axs[0].set_xlabel("range (m)")
    axs[0].set_ylabel("|H - bg| (dB)")
    axs[0].grid(alpha=0.3)
    axs[0].legend(loc="upper right")
    title_top = axs[0].set_title("")

    # bottom: full waveform + cursor
    axs[1].axvspan(0, t[BG_FRAMES - 1], color="tab:blue", alpha=0.08,
                   label="background window")
    axs[1].plot(t, disp_bp, color="tab:gray", lw=0.9, label="filtered (0.1-0.6 Hz)")
    cursor = axs[1].axvline(t[BG_FRAMES], color="r", lw=1.2)
    pt, = axs[1].plot([t[BG_FRAMES]], [disp_bp[BG_FRAMES]], "ro", ms=5)
    axs[1].set_xlim(0, t[-1])
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("chest displacement (mm)")
    axs[1].set_title(
        f"chest displacement @ {rng[target_bin]:.2f} m   |   "
        f"breathing rate ≈ {bpm:.1f} BPM ({peak_f:.2f} Hz)"
    )
    axs[1].grid(alpha=0.3)
    axs[1].legend(loc="upper right")

    plt.tight_layout()

    def update(i):
        line_r.set_ydata(test_mag[i])
        f_idx = BG_FRAMES + i
        ti = f_idx * FRAME_PERIOD_S
        cursor.set_xdata([ti, ti])
        pt.set_data([ti], [disp_bp[f_idx]])
        title_top.set_text(f"frame {f_idx} / {n_frames-1}   t = {ti:5.2f} s")
        return line_r, cursor, pt, title_top

    ani = FuncAnimation(fig, update, frames=test_mag.shape[0],
                        interval=1000 / GIF_FPS, blit=False)
    ani.save("Gifs/" + str(Path(args.in_npy).with_suffix(".gif")), writer=PillowWriter(fps=GIF_FPS))
    print(f"saved {"Gifs/" + str(Path(args.in_npy).with_suffix(".gif"))}  ({test_mag.shape[0]} frames @ {GIF_FPS} fps)")


if __name__ == "__main__":
    main()