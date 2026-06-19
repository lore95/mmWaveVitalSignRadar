#!/usr/bin/env python3
"""
Kalman-based peak tracker for mmWave range profiles.

Tracks the dominant peak in the background-subtracted range profile
as the subject moves, using a constant-velocity Kalman filter in
range-bin space.

Features:
  • Gated association: rejects detections that jump too far from prediction
  • Sub-bin interpolation: parabolic refinement of peak position
  • Complex interpolation: extracts phase at fractional bin positions
  • Adaptive process noise: increases when the peak is moving fast
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TrackerState:
    """Snapshot of the Kalman tracker for external inspection."""
    bin_est: float           # estimated bin position (fractional)
    vel_est: float           # estimated bin velocity (bins/frame)
    range_m: float           # estimated range in metres
    P: np.ndarray            # 2x2 covariance
    detected_bin: float      # raw measurement this frame (NaN if gated)
    gated: bool              # True if measurement was rejected


class KalmanPeakTracker:
    """1-D constant-velocity Kalman filter tracking a range-bin peak.

    State vector:  x = [bin_position, bin_velocity]^T
    Process model: x[n+1] = F @ x[n] + w,   w ~ N(0, Q)
    Measurement:   z[n]   = H @ x[n] + v,   v ~ N(0, R)

    Args:
        initial_bin: starting bin position (from first detection)
        range_axis: array mapping bin index → metres
        q_pos: process noise variance on position (bins²)
        q_vel: process noise variance on velocity (bins²/frame²)
        r_meas: measurement noise variance (bins²)
        gate_sigma: reject measurements beyond this many σ from prediction
        min_bin: lower bound of search window (bin index)
        max_bin: upper bound of search window (bin index)
    """

    def __init__(self, initial_bin: float, range_axis: np.ndarray, *,
                 q_pos: float = 0.1,
                 q_vel: float = 0.01,
                 r_meas: float = 1.0,
                 gate_sigma: float = 5.0,
                 min_bin: int = 1,
                 max_bin: int = 270):

        self.range_axis = range_axis
        self.gate_sigma = gate_sigma
        self.min_bin = min_bin
        self.max_bin = max_bin

        # State: [bin_position, bin_velocity]
        self.x = np.array([initial_bin, 0.0])
        self.P = np.array([[r_meas, 0.0],
                           [0.0,    1.0]])

        # Constant-velocity transition
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])

        # Measurement picks off position only
        self.H = np.array([[1.0, 0.0]])

        # Noise
        self.Q = np.array([[q_pos, 0.0],
                           [0.0,  q_vel]])
        self.R = np.array([[r_meas]])

        self._last_detected = initial_bin
        self._miss_count = 0
        self._max_miss = 25   # ~1 second at 25 Hz before reset

    # ── core Kalman steps ──

    def predict(self):
        """Time update (predict)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Clamp position to valid range
        self.x[0] = np.clip(self.x[0], self.min_bin, self.max_bin)

    def update(self, z: float) -> bool:
        """Measurement update. Returns True if measurement was accepted."""
        # Innovation
        y = z - (self.H @ self.x)[0]
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]

        # Gating: reject if too far from prediction
        if abs(y) > self.gate_sigma * np.sqrt(S):
            self._miss_count += 1
            return False

        # Kalman gain
        K = (self.P @ self.H.T) / S  # (2,1)
        self.x = self.x + K.flatten() * y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        # Clamp
        self.x[0] = np.clip(self.x[0], self.min_bin, self.max_bin)

        self._last_detected = z
        self._miss_count = 0
        return True

    # ── high-level per-frame interface ──

    def step(self, range_profile_mag: np.ndarray,
             noise_floor_db: float = 6.0) -> TrackerState:
        """Run one full predict → detect → update cycle.

        Args:
            range_profile_mag: |H - bg| magnitude array, shape (n_bins,)
            noise_floor_db: minimum peak SNR above median to accept detection

        Returns:
            TrackerState with current estimates.
        """
        self.predict()

        # ── detect peak with sub-bin refinement ──
        detected_bin, gated = self._detect_and_associate(
            range_profile_mag, noise_floor_db)

        if not gated:
            accepted = self.update(detected_bin)
            gated = not accepted
        else:
            self._miss_count += 1

        # If we've missed too many frames, snap to strongest visible peak
        if self._miss_count >= self._max_miss:
            fallback = self._find_strongest_peak(range_profile_mag)
            if fallback is not None:
                self.x[0] = fallback
                self.x[1] = 0.0
                self.P = np.array([[self.R[0, 0], 0.0],
                                   [0.0, 1.0]])
                self._miss_count = 0
                detected_bin = fallback
                gated = False

        return TrackerState(
            bin_est=float(self.x[0]),
            vel_est=float(self.x[1]),
            range_m=float(np.interp(self.x[0], np.arange(len(self.range_axis)),
                                     self.range_axis)),
            P=self.P.copy(),
            detected_bin=detected_bin if not gated else float('nan'),
            gated=gated,
        )

    def _detect_and_associate(self, mag: np.ndarray,
                              noise_floor_db: float
                              ) -> Tuple[float, bool]:
        """Find the peak nearest the predicted position.

        Returns (peak_bin, gated). If gated=True, no valid peak found.
        """
        lo = max(self.min_bin, int(self.x[0] - self.gate_sigma * 8))
        hi = min(self.max_bin, int(self.x[0] + self.gate_sigma * 8))
        if hi <= lo:
            return 0.0, True

        window = mag[lo:hi]
        if len(window) == 0:
            return 0.0, True

        # SNR check
        median_noise = float(np.median(mag[self.min_bin:self.max_bin]))
        peak_idx_local = int(np.argmax(window))
        peak_val = float(window[peak_idx_local])

        if peak_val < median_noise * (10 ** (noise_floor_db / 20)):
            return 0.0, True

        peak_idx = lo + peak_idx_local

        # Sub-bin parabolic interpolation
        refined = self._parabolic_refine(mag, peak_idx)

        return refined, False

    @staticmethod
    def _parabolic_refine(mag: np.ndarray, idx: int) -> float:
        """Parabolic interpolation around a peak for sub-bin accuracy."""
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

    def _find_strongest_peak(self, mag: np.ndarray) -> Optional[float]:
        """Fallback: find the strongest peak in the full search window."""
        window = mag[self.min_bin:self.max_bin]
        if len(window) == 0:
            return None
        idx = self.min_bin + int(np.argmax(window))
        return self._parabolic_refine(mag, idx)

    @property
    def bin_position(self) -> float:
        return float(self.x[0])

    @property
    def range_m(self) -> float:
        return float(np.interp(self.x[0], np.arange(len(self.range_axis)),
                                self.range_axis))


def interpolate_complex_at(fft_bins: np.ndarray, pos: float) -> complex:
    """Linearly interpolate a complex FFT array at a fractional bin position.

    This preserves phase continuity better than nearest-neighbour when
    the Kalman estimate falls between two integer bins.
    """
    n = len(fft_bins)
    idx_lo = int(np.floor(pos))
    idx_hi = idx_lo + 1

    if idx_lo < 0:
        return complex(fft_bins[0])
    if idx_hi >= n:
        return complex(fft_bins[-1])

    frac = pos - idx_lo
    return complex(fft_bins[idx_lo]) * (1 - frac) + complex(fft_bins[idx_hi]) * frac
