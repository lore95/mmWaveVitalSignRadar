#!/usr/bin/env python3
"""
Multi-target track manager for mmWave range profiles.

Detects multiple peaks in the background-subtracted range profile,
associates them to existing Kalman-tracked targets, manages track
creation/deletion lifecycle, and maintains independent phase/BPM
state per track.

Designed for 1TX × 1RX (range-only, no angle).  Two people at the
same range but different angles will merge into one track.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from scipy.signal import find_peaks, butter, sosfiltfilt

from kalman_tracker import KalmanPeakTracker, interpolate_complex_at


# ═══════════════════════════════════════════════════════════════════════
# Per-track state
# ═══════════════════════════════════════════════════════════════════════

# Colours assigned to tracks in order of creation
TRACK_COLORS = [
    "#00ff99",   # green
    "#ff79c6",   # pink
    "#feca57",   # yellow
    "#0ff",      # cyan
    "#ff6361",   # coral
    "#a29bfe",   # lavender
    "#fd79a8",   # hot pink
    "#55efc4",   # mint
]

BP_LO, BP_HI = 0.1, 0.6


def _make_bandpass(lo, hi, fs, order=4):
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def _estimate_bpm(signal_seg, fs):
    if len(signal_seg) < 64:
        return 0.0
    sig = signal_seg - signal_seg.mean()
    f = np.fft.rfftfreq(len(sig), d=1 / fs)
    mag = np.abs(np.fft.rfft(sig))
    band = (f >= BP_LO) & (f <= BP_HI)
    if not band.any():
        return 0.0
    return float(f[band][np.argmax(mag[band])]) * 60.0


@dataclass
class Track:
    """Independent state for one tracked person."""
    track_id: int
    kalman: KalmanPeakTracker
    color: str

    # Phase tracking (each track has its own unwrapper)
    prev_phase: float = 0.0

    # History buffers
    phase_hist: deque = field(default_factory=lambda: deque(maxlen=750))
    disp_hist: deque = field(default_factory=lambda: deque(maxlen=750))
    time_hist: deque = field(default_factory=lambda: deque(maxlen=750))

    # BPM
    bpm: float = 0.0
    _last_bpm_time: float = 0.0   # wall time of last BPM recomputation

    # Lifecycle
    miss_count: int = 0
    hit_count: int = 0
    confirmed: bool = False
    age: int = 0  # total frames since creation

    @property
    def bin_position(self) -> float:
        return self.kalman.bin_position

    @property
    def range_m(self) -> float:
        return self.kalman.range_m

    def process_phase(self, rfft_avg: np.ndarray, t_rel: float,
                      lambda_m: float, fps: float, sos_bp,
                      bpm_window_s: float = 30.0,
                      bpm_refresh_s: float = 5.0):
        """Extract phase at the tracked bin, unwrap, filter, estimate BPM.

        Phase extraction and displacement filtering run every frame.
        BPM spectral estimate only recomputes every bpm_refresh_s seconds.
        """
        tracked_bin = self.kalman.bin_position
        z = interpolate_complex_at(rfft_avg, tracked_bin)
        ang = np.angle(z)

        # Manual unwrap relative to this track's own prev_phase
        diff = ang - self.prev_phase
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        self.prev_phase = self.prev_phase + diff
        unwrapped = self.prev_phase

        self.phase_hist.append(unwrapped)
        self.time_hist.append(t_rel)

        # Phase → displacement (runs every frame for smooth waveform)
        if len(self.phase_hist) > 10:
            ph = np.array(self.phase_hist)
            ts = np.array(self.time_hist)
            a, b = np.polyfit(ts, ph, 1)
            ph_dt = ph - (a * ts + b)
            disp_mm = -lambda_m * ph_dt / (4 * np.pi) * 1000.0

            if len(disp_mm) > 27:
                try:
                    disp_filt = sosfiltfilt(sos_bp, disp_mm)
                except Exception:
                    disp_filt = disp_mm
            else:
                disp_filt = disp_mm

            self.disp_hist.clear()
            self.disp_hist.extend(disp_filt)

            # BPM estimate — only refresh every bpm_refresh_s seconds
            import time as _time
            now = _time.monotonic()
            if now - self._last_bpm_time >= bpm_refresh_s:
                win_n = min(len(disp_filt), int(bpm_window_s * fps))
                self.bpm = _estimate_bpm(disp_filt[-win_n:], fps)
                self._last_bpm_time = now


# ═══════════════════════════════════════════════════════════════════════
# Track manager
# ═══════════════════════════════════════════════════════════════════════

class MultiTrackManager:
    """Manages multiple independently-tracked targets.

    Per-frame flow:
        1. detect_peaks()          → list of peak positions
        2. associate()             → match peaks to existing tracks
        3. update matched tracks   → Kalman update + phase extraction
        4. handle unmatched peaks  → tentative new tracks
        5. handle unmatched tracks → increment miss, delete if stale

    Args:
        range_axis: array of range (m) per bin
        lambda_m: radar wavelength (m)
        fps: frame rate (Hz)
        min_bin, max_bin: search window in bin indices
        max_tracks: maximum simultaneous tracks
        confirm_frames: consecutive hits before a track is confirmed
        delete_frames: consecutive misses before a track is deleted
        min_peak_separation_bins: minimum distance between detected peaks
        min_track_separation_bins: minimum distance between two tracks
        snr_threshold_db: minimum peak SNR above noise floor
        kalman_kwargs: passed to KalmanPeakTracker constructor
    """

    def __init__(self, range_axis: np.ndarray, lambda_m: float, fps: float,
                 min_bin: int = 5, max_bin: int = 152,
                 max_tracks: int = 4,
                 confirm_frames: int = 3,
                 delete_frames: int = 50,
                 min_peak_separation_bins: int = 10,
                 min_track_separation_bins: int = 8,
                 snr_threshold_db: float = 6.0,
                 bpm_window_s: float = 30.0,
                 bpm_refresh_s: float = 5.0,
                 **kalman_kwargs):

        self.range_axis = range_axis
        self.lambda_m = lambda_m
        self.fps = fps
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.max_tracks = max_tracks
        self.confirm_frames = confirm_frames
        self.delete_frames = delete_frames
        self.min_peak_sep = min_peak_separation_bins
        self.min_track_sep = min_track_separation_bins
        self.snr_db = snr_threshold_db
        self.bpm_window_s = bpm_window_s
        self.bpm_refresh_s = bpm_refresh_s
        self.kalman_kwargs = kalman_kwargs

        self.tracks: List[Track] = []
        self._next_id = 0
        self._color_idx = 0

        # Per-bin background statistics (set by set_background_stats)
        self._bg_std: Optional[np.ndarray] = None  # per-bin σ from calibration
        self._cfar_k: float = 4.0  # detection threshold = k × σ per bin

        # Bandpass filter (shared, all tracks use same fs)
        self.sos_bp = _make_bandpass(BP_LO, BP_HI, fps)

    def set_background_stats(self, bg_std: np.ndarray, cfar_k: float = 4.0):
        """Set per-bin noise statistics from calibration for CFAR detection.

        Args:
            bg_std: standard deviation of |H_bg[k]| per bin, shape (n_bins,)
                    computed from the calibration frames with empty environment.
            cfar_k: detection threshold multiplier. A peak at bin k must
                    exceed cfar_k × bg_std[k] after background subtraction
                    to be considered a detection.  Higher = fewer false alarms.
        """
        self._bg_std = bg_std.copy()
        self._cfar_k = cfar_k
        # Floor the std at a small value to avoid division issues in quiet bins
        self._bg_std = np.maximum(self._bg_std, 1e-6)
        print(f"[TRACK] CFAR enabled: k={cfar_k:.1f}, "
              f"mean σ={float(np.mean(bg_std[self.min_bin:self.max_bin])):.4f}, "
              f"max σ={float(np.max(bg_std[self.min_bin:self.max_bin])):.4f}")

    # ── peak detection ──

    def detect_peaks(self, mag: np.ndarray) -> List[float]:
        """Find all peaks in the magnitude profile within the search window.

        When per-bin background statistics are available (via set_background_stats),
        uses CFAR detection: each bin has its own threshold = cfar_k × σ[k].
        Falls back to global 75th-percentile SNR gating otherwise.

        Returns list of sub-bin-refined peak positions.
        """
        search_region = mag[self.min_bin:self.max_bin]
        if len(search_region) == 0:
            return []

        if self._bg_std is not None:
            # ── CFAR detection: per-bin adaptive threshold ──
            # Threshold array for the search region
            thresh_region = self._cfar_k * self._bg_std[self.min_bin:self.max_bin]

            # Mask bins below their individual threshold
            above_thresh = search_region > thresh_region

            if not above_thresh.any():
                return []

            # Use threshold as the minimum height for find_peaks
            # Global height = minimum of per-bin thresholds (find_peaks needs scalar)
            # but we post-filter by per-bin threshold
            min_height = float(np.min(thresh_region[thresh_region > 0]))

            indices, properties = find_peaks(
                search_region,
                distance=self.min_peak_sep,
                height=min_height,
                prominence=min_height,
            )

            # Post-filter: each peak must exceed its own bin's threshold
            valid = []
            for idx_local in indices:
                if search_region[idx_local] > thresh_region[idx_local]:
                    valid.append(idx_local)
            indices = valid

        else:
            # ── Fallback: global noise floor ──
            noise_floor = float(np.percentile(search_region, 75))
            snr_threshold = noise_floor * (10 ** (self.snr_db / 20))

            indices_arr, properties = find_peaks(
                search_region,
                distance=self.min_peak_sep,
                height=snr_threshold,
                prominence=snr_threshold,
            )
            indices = list(indices_arr)

        # Convert to global bin indices and refine
        peaks = []
        for idx_local in indices:
            idx_global = self.min_bin + int(idx_local)
            refined = self._parabolic_refine(mag, idx_global)
            peaks.append(refined)

        return peaks

    @staticmethod
    def _parabolic_refine(mag: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(mag) - 1:
            return float(idx)
        alpha = float(mag[idx - 1])
        beta = float(mag[idx])
        gamma = float(mag[idx + 1])
        denom = 2.0 * (2.0 * beta - alpha - gamma)
        if abs(denom) < 1e-12:
            return float(idx)
        offset = (alpha - gamma) / denom
        return float(idx) + np.clip(offset, -0.5, 0.5)

    # ── association ──

    def associate(self, peaks: List[float]
                  ) -> Tuple[List[Tuple[Track, float]],
                             List[float],
                             List[Track]]:
        """Match detected peaks to existing tracks by nearest-neighbour.

        Returns:
            matched:  list of (track, peak_position) pairs
            unmatched_peaks: peaks with no nearby track
            unmatched_tracks: tracks with no nearby peak
        """
        if not self.tracks and not peaks:
            return [], [], []

        matched = []
        used_peaks = set()
        used_tracks = set()

        # Build predicted positions
        predictions = [(t, t.kalman.bin_position) for t in self.tracks]

        # Greedy nearest-neighbour (sort by distance)
        pairs = []
        for ti, (track, pred) in enumerate(predictions):
            for pi, peak in enumerate(peaks):
                dist = abs(pred - peak)
                pairs.append((dist, ti, pi, track, peak))
        pairs.sort(key=lambda x: x[0])

        # Gate: peak must be within track's Kalman uncertainty or min_peak_sep,
        # whichever is smaller — prevents noise from keeping a dying track alive
        for dist, ti, pi, track, peak in pairs:
            if ti in used_tracks or pi in used_peaks:
                continue
            # Use Kalman position uncertainty for gating
            pos_sigma = np.sqrt(track.kalman.P[0, 0])
            gate = min(self.min_peak_sep, pos_sigma * self.kalman_kwargs.get("gate_sigma", 5.0))
            gate = max(gate, 3.0)  # floor at 3 bins
            if dist > gate:
                continue
            matched.append((track, peak))
            used_tracks.add(ti)
            used_peaks.add(pi)

        unmatched_peaks = [p for i, p in enumerate(peaks) if i not in used_peaks]
        unmatched_tracks = [t for i, (t, _) in enumerate(predictions)
                            if i not in used_tracks]

        return matched, unmatched_peaks, unmatched_tracks

    # ── lifecycle ──

    def _spawn_track(self, peak_bin: float, rfft_avg: np.ndarray) -> Track:
        """Create a new tentative track at the given bin."""
        kalman = KalmanPeakTracker(
            initial_bin=peak_bin,
            range_axis=self.range_axis,
            min_bin=self.min_bin,
            max_bin=self.max_bin,
            **self.kalman_kwargs,
        )

        color = TRACK_COLORS[self._color_idx % len(TRACK_COLORS)]
        self._color_idx += 1

        track = Track(
            track_id=self._next_id,
            kalman=kalman,
            color=color,
            prev_phase=float(np.angle(interpolate_complex_at(rfft_avg, peak_bin))),
            hit_count=1,
        )
        self._next_id += 1
        return track

    def _is_too_close_to_existing(self, peak_bin: float) -> bool:
        """Check if a peak is too close to any existing track (confirmed or tentative)."""
        for t in self.tracks:
            if abs(t.bin_position - peak_bin) < self.min_track_sep:
                return True
        return False

    # ── main per-frame entry point ──

    def step(self, mag: np.ndarray, rfft_avg: np.ndarray, t_rel: float):
        """Run the full multi-track pipeline for one radar frame.

        Args:
            mag: background-subtracted magnitude profile (n_bins,)
            rfft_avg: complex range FFT averaged across chirps (n_bins,)
            t_rel: relative timestamp (seconds since epoch)
        """
        # 1. Detect all peaks
        peaks = self.detect_peaks(mag)

        # 2. Predict all existing tracks
        for t in self.tracks:
            t.kalman.predict()
            t.age += 1

        # 3. Associate peaks to tracks
        matched, unmatched_peaks, unmatched_tracks = self.associate(peaks)

        # 4. Update matched tracks
        for track, peak in matched:
            track.kalman.update(peak)
            track.miss_count = 0
            track.hit_count += 1

            if not track.confirmed and track.hit_count >= self.confirm_frames:
                track.confirmed = True
                print(f"[TRACK] #{track.track_id} confirmed @ "
                      f"{track.range_m:.2f} m")

            track.process_phase(rfft_avg, t_rel, self.lambda_m,
                                self.fps, self.sos_bp, self.bpm_window_s,
                                self.bpm_refresh_s)

        # 5. Handle unmatched tracks (missed detection)
        for track in unmatched_tracks:
            track.miss_count += 1
            # Still extract phase at predicted position (coast)
            if track.confirmed:
                track.process_phase(rfft_avg, t_rel, self.lambda_m,
                                    self.fps, self.sos_bp, self.bpm_window_s,
                                    self.bpm_refresh_s)

        # 6. Delete stale tracks BEFORE spawning new ones
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks
                       if t.miss_count < self.delete_frames]
        deleted = before - len(self.tracks)
        if deleted:
            print(f"[TRACK] deleted {deleted} stale track(s), "
                  f"{len(self.tracks)} active")

        # 7. Handle unmatched peaks (potential new targets)
        for peak in unmatched_peaks:
            if len(self.tracks) >= self.max_tracks:
                break
            if not self._is_too_close_to_existing(peak):
                new_track = self._spawn_track(peak, rfft_avg)
                self.tracks.append(new_track)

    # ── accessors ──

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.confirmed]

    @property
    def all_tracks(self) -> List[Track]:
        return list(self.tracks)

    @property
    def num_confirmed(self) -> int:
        return sum(1 for t in self.tracks if t.confirmed)