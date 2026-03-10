"""
Driver model module for lap time simulation.

Provides pure functions and dataclasses that compute driver input channels
(throttle, brake, steering, gear, RPM) from kinematic simulation outputs.

References
----------
Segers, J. (2014). Analysis Techniques for Racecar Data Acquisition,
  2nd Ed. SAE International.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..vehicle.parameters import VehicleParams


@dataclass
class DriverInputs:
    """Driver input channels for one complete simulation lap."""
    throttle_pct: np.ndarray
    brake_pct: np.ndarray
    steering_deg: np.ndarray
    gear: np.ndarray
    rpm: np.ndarray


@dataclass
class DriverModel:
    """Configurable driver model parameters."""
    steering_ratio: float = 15.0
    throttle_lag: float = 0.05
    brake_lag: float = 0.03
    smooth_window: int = 1


def compute_throttle_brake(
    ax_long: np.ndarray,
    smooth_window: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive throttle_pct and brake_pct from longitudinal acceleration."""
    a_pos = np.clip(ax_long, 0.0, None)
    a_neg = np.clip(-ax_long, 0.0, None)
    peak_accel = max(float(np.max(a_pos)), 1e-6)
    peak_brake = max(float(np.max(a_neg)), 1e-6)
    throttle = np.clip(a_pos / peak_accel * 100.0, 0.0, 100.0)
    brake    = np.clip(a_neg / peak_brake * 100.0, 0.0, 100.0)
    if smooth_window > 1:
        kernel   = np.ones(smooth_window) / smooth_window
        throttle = np.clip(np.convolve(throttle, kernel, mode='same'), 0.0, 100.0)
        brake    = np.clip(np.convolve(brake,    kernel, mode='same'), 0.0, 100.0)
    return throttle, brake


def compute_gear(
    v_ms: np.ndarray,
    params: VehicleParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Select optimal gear at each track point to maximise drive force."""
    n = len(v_ms)
    gear_arr = np.ones(n, dtype=int)
    rpm_arr  = np.zeros(n)

    gear_ratios  = params.transmission.gear_ratios
    final_drive  = params.transmission.final_drive_ratio
    r_wheel      = params.tire.wheel_radius
    rpm_idle     = params.engine.rpm_idle
    rpm_max      = params.engine.rpm_max
    rpm_min_opt  = rpm_idle * 1.5
    rpm_max_opt  = rpm_max  * 0.90
    n_gears      = params.transmission.num_gears

    use_map   = len(params.engine.torque_curve_rpm) > 0
    t_map_rpm = np.array(params.engine.torque_curve_rpm) if use_map else None
    t_map_nm  = np.array(params.engine.torque_curve_nm)  if use_map else None

    def _torque(rpm: float) -> float:
        if use_map:
            return float(np.interp(
                np.clip(rpm, t_map_rpm[0], t_map_rpm[-1]), t_map_rpm, t_map_nm
            ))
        rpm_peak = 1300.0
        T_max = params.engine.max_torque
        if rpm < rpm_idle:
            return 0.0
        elif rpm <= rpm_peak:
            return T_max * (rpm - rpm_idle) / (rpm_peak - rpm_idle)
        elif rpm <= rpm_max:
            return T_max * float(np.exp(-0.0015 * (rpm - rpm_peak) ** 1.2))
        return 0.0

    for i, v in enumerate(v_ms):
        best_gear, best_force, best_rpm = 1, -1.0, rpm_idle
        for g in range(1, n_gears + 1):
            ratio = gear_ratios[g - 1] * final_drive
            rpm = (max(v, 0.01) / r_wheel) * ratio * 60.0 / (2.0 * np.pi)
            if rpm > rpm_max:
                continue
            rpm = max(rpm, rpm_idle)
            T = _torque(rpm)
            F = T * ratio / r_wheel
            in_band = rpm_min_opt <= rpm <= rpm_max_opt
            if in_band and F > best_force:
                best_force = F
                best_gear  = g
                best_rpm   = rpm
            elif best_force < 0.0:
                if F > best_force:
                    best_force = F
                    best_gear  = g
                    best_rpm   = rpm
        gear_arr[i] = best_gear
        rpm_arr[i]  = best_rpm

    return gear_arr, rpm_arr


def compute_steering(
    radius: np.ndarray,
    wheelbase: float,
    steering_ratio: float = 15.0,
) -> np.ndarray:
    """Estimate steering wheel angle from Ackermann geometry."""
    delta_rad = wheelbase / np.maximum(radius, 1.0)
    return np.degrees(delta_rad) * steering_ratio


def compute_driver_inputs(
    v_ms: np.ndarray,
    ax_long: np.ndarray,
    radius: np.ndarray,
    params: VehicleParams,
    driver: Optional[DriverModel] = None,
) -> DriverInputs:
    """Compute all driver input channels from kinematic solver outputs."""
    if driver is None:
        driver = DriverModel()
    throttle, brake = compute_throttle_brake(ax_long, driver.smooth_window)
    gear, rpm       = compute_gear(v_ms, params)
    steering        = compute_steering(
        radius,
        wheelbase=params.mass_geometry.wheelbase,
        steering_ratio=driver.steering_ratio,
    )
    return DriverInputs(
        throttle_pct=throttle,
        brake_pct=brake,
        steering_deg=steering,
        gear=gear,
        rpm=rpm,
    )
