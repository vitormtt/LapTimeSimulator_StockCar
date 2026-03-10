# src/simulation/lap_time_solver.py
"""
Simulador de Lap Time — Solver principal

Ponto de entrada principal: run_simulation(config, vehicle_params, circuit)
Legacy entry point preservado: run_bicycle_model(params_dict, circuit, config)

Modos suportados (via SimulationMode):
  QUALIFYING    — volta de classificação a partir de velocidade de equilíbrio
  FLYING_LAP    — volta com velocidade de entrada prescrita (v_entry_kmh)
  STANDING_START— largada parada com modelo de patinagem e rampa de embreagem

O solver aplica VehicleSetup automaticamente antes de resolver,
modificando parâmetros de aero, pneus e freio conforme configurado.

Output: SimulationResult com canais de telemetria alinhados ao
nomenclador Pi Toolbox / MoTeC (Porsche Carrera Cup Brasil).

Referencias
-----------
Brayshaw & Harrison (2005). A quasi steady state approach to race
  car lap simulation. Proc. IMechE Part D, 219(3), 383-394.
Segers, J. (2014). Analysis Techniques for Racecar Data Acquisition,
  2nd Ed. SAE International.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.simulation.simulation_modes import SimulationConfig, SimulationMode
from src.vehicle.parameters import VehicleParams
from src.vehicle.setup import apply_setup_to_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy local VehicleParams dataclass (kept for run_bicycle_model compat)
# ---------------------------------------------------------------------------

@dataclass
class _LegacyVehicleParams:
    """Internal flat params used by legacy run_bicycle_model only."""
    m: float = 5000.0
    lf: float = 2.1
    lr: float = 2.3
    h_cg: float = 1.1
    Cf: float = 120000.0
    Cr: float = 120000.0
    mu: float = 1.1
    r_wheel: float = 0.65
    P_max: float = 600000.0
    T_max: float = 3700.0
    rpm_max: float = 2800.0
    rpm_idle: float = 800.0
    n_gears: int = 12
    gear_ratios: list = None
    final_drive: float = 5.33
    max_decel: float = 7.5
    Cx: float = 0.85
    A_front: float = 8.7
    Cl: float = 0.0

    def __post_init__(self):
        if self.gear_ratios is None:
            self.gear_ratios = [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1,
                                 1.6, 1.25, 1.0, 0.78]
        self.L = self.lf + self.lr
        self.Iz = self.m * (self.lf**2 + self.lr**2) / 2


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Output container for a completed simulation run.

    All array channels are 1-D numpy arrays of length n (number of track
    points). Scalar KPIs are pre-computed at construction.

    Attributes
    ----------
    lap_time : float
        Total simulated lap time [s].
    mode : SimulationMode
        Mode used for this simulation.
    setup_name : str
        Name tag of the VehicleSetup applied.
    distance : np.ndarray
        Cumulative distance along track [m].
    time : np.ndarray
        Cumulative lap time at each point [s].
    v_kmh : np.ndarray
        Speed [km/h].
    ax_long_g : np.ndarray
        Longitudinal acceleration [g]. Positive = acceleration.
    ay_lat_g : np.ndarray
        Lateral acceleration [g]. Positive = left.
    throttle_pct : np.ndarray
        Throttle demand [0–100 %].
    brake_pct : np.ndarray
        Brake demand [0–100 %].
    steering_deg : np.ndarray
        Estimated steering wheel angle [deg].
    gear : np.ndarray
        Engaged gear (integer).
    rpm : np.ndarray
        Engine RPM.
    radius : np.ndarray
        Track corner radius at each point [m].
    temp_tyre_c : np.ndarray
        Tyre bulk temperature [degC].
    tyre_pressure_bar : np.ndarray
        Hot tyre pressure estimate [bar].
    fuel_used_l : np.ndarray
        Cumulative fuel consumption [L].
    """
    lap_time: float
    mode: SimulationMode
    setup_name: str

    distance: np.ndarray
    time: np.ndarray
    v_kmh: np.ndarray
    ax_long_g: np.ndarray
    ay_lat_g: np.ndarray
    throttle_pct: np.ndarray
    brake_pct: np.ndarray
    steering_deg: np.ndarray
    gear: np.ndarray
    rpm: np.ndarray
    radius: np.ndarray
    temp_tyre_c: np.ndarray
    tyre_pressure_bar: np.ndarray
    fuel_used_l: np.ndarray

    # Raw (m/s²) versions kept for internal use
    _a_long_ms2: np.ndarray = field(repr=False, default=None)
    _a_lat_ms2: np.ndarray = field(repr=False, default=None)

    # -----------------------------------------------------------------------
    # KPI properties
    # -----------------------------------------------------------------------

    @property
    def avg_speed_kmh(self) -> float:
        return float(np.mean(self.v_kmh))

    @property
    def max_speed_kmh(self) -> float:
        return float(np.max(self.v_kmh))

    @property
    def peak_lat_g(self) -> float:
        return float(np.max(np.abs(self.ay_lat_g)))

    @property
    def peak_brake_g(self) -> float:
        return float(np.min(self.ax_long_g))

    @property
    def peak_accel_g(self) -> float:
        return float(np.max(self.ax_long_g))

    @property
    def time_wot_pct(self) -> float:
        """Percentage of lap with throttle >= 95%."""
        return float(np.mean(self.throttle_pct >= 95.0) * 100.0)

    @property
    def time_braking_pct(self) -> float:
        """Percentage of lap with brake > 5%."""
        return float(np.mean(self.brake_pct > 5.0) * 100.0)

    @property
    def fuel_total_l(self) -> float:
        return float(self.fuel_used_l[-1])

    @property
    def final_tyre_temp_c(self) -> float:
        return float(self.temp_tyre_c[-1])

    @property
    def final_tyre_pressure_bar(self) -> float:
        return float(self.tyre_pressure_bar[-1])

    def to_dataframe(self) -> pd.DataFrame:
        """Export all channels to a tidy DataFrame (MoTeC/Pi Toolbox compatible)."""
        return pd.DataFrame({
            "distance_m":     self.distance,
            "lap_time_s":     self.time,
            "v_kmh":          self.v_kmh,
            "ax_long_g":      self.ax_long_g,
            "ay_lat_g":       self.ay_lat_g,
            "throttle_pct":   self.throttle_pct,
            "brake_pct":      self.brake_pct,
            "steering_deg":   self.steering_deg,
            "gear":           self.gear,
            "rpm":            self.rpm,
            "radius_m":       self.radius,
            "temp_tyre_c":    self.temp_tyre_c,
            "tyre_press_bar": self.tyre_pressure_bar,
            "fuel_used_l":    self.fuel_used_l,
        })

    def save_csv(self, path: str) -> None:
        """Save telemetry to CSV. Filename format compatible with existing app."""
        self.to_dataframe().to_csv(path, index=False)
        logger.info(f"[OK] Telemetria salva em: {path}")

    def log_kpis(self) -> None:
        """Log performance KPIs to INFO."""
        logger.info(
            f"[RESULT] [{self.mode.name}] Setup='{self.setup_name}' | "
            f"Lap={self.lap_time:.2f}s | "
            f"V_avg={self.avg_speed_kmh:.1f} km/h | "
            f"V_max={self.max_speed_kmh:.1f} km/h | "
            f"Peak_lat={self.peak_lat_g:.2f}g | "
            f"WOT={self.time_wot_pct:.1f}% | "
            f"Braking={self.time_braking_pct:.1f}% | "
            f"T_tyre={self.final_tyre_temp_c:.1f}\u00b0C | "
            f"Fuel={self.fuel_total_l:.2f}L"
        )


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _build_flat_params(vp: VehicleParams) -> _LegacyVehicleParams:
    """Convert structured VehicleParams to flat legacy struct for the solver loop."""
    d = vp.to_solver_dict()
    p = _LegacyVehicleParams(**{k: v for k, v in d.items()
                                 if k in _LegacyVehicleParams.__dataclass_fields__})
    return p


def _compute_track_geometry(circuit) -> tuple:
    """Compute ds, s, curvature radius arrays from circuit centerline."""
    x = circuit.centerline_x
    y = circuit.centerline_y
    n = len(x)

    ds = np.zeros(n)
    ds[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.cumsum(ds)

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
    radius = np.where(np.abs(curvature) > 1e-6, 1.0 / np.abs(curvature), 1e6)
    radius = np.clip(radius, 10.0, 1e6)

    return x, y, n, ds, s, radius


def _torque_curve(rpm: float, p: _LegacyVehicleParams) -> float:
    """Engine torque [N·m] at given RPM. Diesel heavy truck character."""
    rpm_torque_max = 1300.0
    if rpm < p.rpm_idle:
        return 0.0
    elif rpm <= rpm_torque_max:
        return p.T_max * (rpm - p.rpm_idle) / (rpm_torque_max - p.rpm_idle)
    elif rpm <= p.rpm_max:
        return p.T_max * np.exp(-0.0015 * (rpm - rpm_torque_max) ** 1.2)
    else:
        return 0.0


def _torque_curve_interp(
    rpm: float,
    torque_curve_rpm: list,
    torque_curve_nm: list,
    rpm_max: float,
) -> float:
    """
    Interpolated torque from VehicleParams engine map.

    Used when VehicleParams has a non-empty torque_curve_rpm list
    (e.g. Porsche flat-6 map). Falls back to _torque_curve if list empty.
    """
    if not torque_curve_rpm:
        return 0.0
    rpm_c = float(np.clip(rpm, torque_curve_rpm[0], torque_curve_rpm[-1]))
    return float(np.interp(rpm_c, torque_curve_rpm, torque_curve_nm))


def _select_gear_optimal(v: float, p: _LegacyVehicleParams) -> int:
    """Select gear that maximises drive force within RPM range."""
    rpm_min_opt = p.rpm_idle * 1.5
    rpm_max_opt = p.rpm_max * 0.90
    best_gear, best_force = 1, -1.0
    for gear in range(1, p.n_gears + 1):
        ratio_total = p.gear_ratios[gear - 1] * p.final_drive
        rpm = (v / max(p.r_wheel, 0.01)) * ratio_total * 60.0 / (2 * np.pi)
        if rpm > p.rpm_max:
            continue
        rpm = max(rpm, p.rpm_idle)
        T = _torque_curve(rpm, p)
        F = T * ratio_total / p.r_wheel
        if rpm_min_opt <= rpm <= rpm_max_opt:
            if F > best_force:
                best_force = F
                best_gear = gear
        elif best_force < 0:
            best_gear = gear
    return best_gear


def _get_rpm(v: float, gear: int, p: _LegacyVehicleParams) -> float:
    """Engine RPM at speed v in given gear."""
    if gear < 1 or gear > p.n_gears:
        return p.rpm_idle
    ratio_total = p.gear_ratios[gear - 1] * p.final_drive
    rpm = (v / max(p.r_wheel, 0.01)) * ratio_total * 60.0 / (2 * np.pi)
    return float(np.clip(rpm, p.rpm_idle, p.rpm_max))


def _driver_inputs_from_accel(
    a_long: np.ndarray,
    v_kmh: np.ndarray,
    v_max_kmh: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive throttle_pct and brake_pct from longitudinal acceleration.

    Simple inverse model:
    - throttle proportional to positive ax normalised by peak accel
    - brake proportional to negative ax normalised by peak decel
    """
    a_pos = np.clip(a_long, 0, None)
    a_neg = np.clip(-a_long, 0, None)
    a_max_accel = max(float(np.max(a_pos)), 1e-6)
    a_max_brake  = max(float(np.max(a_neg)), 1e-6)
    throttle = np.clip((a_pos / a_max_accel) * 100.0, 0.0, 100.0)
    brake    = np.clip((a_neg / a_max_brake) * 100.0, 0.0, 100.0)
    return throttle, brake


def _steering_from_radius(
    radius: np.ndarray,
    v_ms: np.ndarray,
    wheelbase: float,
    steering_ratio: float = 15.0,
) -> np.ndarray:
    """
    Estimate steering wheel angle from Ackermann geometry.

    delta_wheel = L / R  (small angle, [rad])
    steering_wheel = delta_wheel * steering_ratio  [deg]
    """
    delta_rad = wheelbase / np.maximum(radius, 1.0)
    # Sign from lateral accel direction (positive curvature = left turn)
    delta_deg = np.degrees(delta_rad) * steering_ratio
    return delta_deg


# ---------------------------------------------------------------------------
# Core GGV solver (shared by QUALIFYING and FLYING_LAP)
# ---------------------------------------------------------------------------

def _run_ggv_solver(
    p: _LegacyVehicleParams,
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    ds: np.ndarray,
    s: np.ndarray,
    radius: np.ndarray,
    mu: float,
    v0: float,
    fuel_per_km: float,
    temp_ini: float,
    p_tyre_cold: float,
    torque_map_rpm: list,
    torque_map_nm: list,
) -> dict:
    """
    GGV forward + backward pass solver.

    Parameters
    ----------
    p : _LegacyVehicleParams
        Flat solver parameters.
    v0 : float
        Initial vehicle speed [m/s].
    fuel_per_km : float
        Fuel consumption [L/km].
    temp_ini : float
        Initial tyre temperature [degC].
    p_tyre_cold : float
        Cold tyre pressure [bar].
    torque_map_rpm / torque_map_nm : list
        Engine torque map for interpolation. If empty, uses diesel model.
    """
    g = 9.81
    rho = 1.225

    v_profile    = np.zeros(n)
    a_long       = np.zeros(n)
    a_lat        = np.zeros(n)
    gear_profile = np.ones(n, dtype=int)
    rpm_profile  = np.zeros(n)
    temp_tyre    = np.ones(n) * temp_ini
    fuel_acum    = np.zeros(n)

    # ---- FORWARD PASS ----
    v_profile[0] = v0
    gear_profile[0] = _select_gear_optimal(v0, p) if v0 > 0 else 1

    for i in range(1, n):
        v_prev = v_profile[i - 1]
        gear   = _select_gear_optimal(v_prev, p)
        gear_profile[i] = gear

        rpm = _get_rpm(v_prev, gear, p)
        rpm_profile[i - 1] = rpm

        # Torque — use interpolated map if available, else diesel model
        if torque_map_rpm:
            T_engine = _torque_curve_interp(rpm, torque_map_rpm, torque_map_nm, p.rpm_max)
        else:
            T_engine = _torque_curve(rpm, p)

        ratio_total = p.gear_ratios[gear - 1] * p.final_drive
        F_traction  = T_engine * ratio_total / p.r_wheel

        F_drag      = 0.5 * rho * p.Cx * p.A_front * v_prev ** 2
        F_downforce = 0.5 * rho * abs(p.Cl) * p.A_front * v_prev ** 2
        F_normal    = p.m * g + F_downforce

        # Traction limited by friction circle
        v_lat_max = np.sqrt(mu * g * radius[i])
        a_lat_cur = v_prev ** 2 / max(radius[i], 1.0)
        F_lat_used = p.m * a_lat_cur
        F_trac_grip = np.sqrt(max((mu * F_normal) ** 2 - F_lat_used ** 2, 0.0))
        F_traction  = min(F_traction, F_trac_grip)

        a = (F_traction - F_drag) / p.m
        a_long[i - 1] = a

        if ds[i] > 0:
            v_possible    = np.sqrt(max(0.0, v_prev ** 2 + 2 * a * ds[i]))
            v_profile[i]  = min(v_possible, v_lat_max)
        else:
            v_profile[i] = v_prev

        # Tyre thermal model (simplified)
        a_total = np.sqrt(a ** 2 + (v_prev ** 2 / max(radius[i], 1.0)) ** 2)
        temp_tyre[i] = temp_tyre[i - 1] + 0.05 * a_total

    # ---- BACKWARD PASS (braking) ----
    for i in reversed(range(n - 1)):
        v_next      = v_profile[i + 1]
        a_lat_next  = v_next ** 2 / max(radius[i + 1], 1.0)
        a_avail     = mu * g
        a_decel_max = np.sqrt(max(0.0, a_avail ** 2 - a_lat_next ** 2))
        a_decel_max = min(a_decel_max, p.max_decel)
        if ds[i + 1] > 0:
            v_brake_limit = np.sqrt(v_next ** 2 + 2 * a_decel_max * ds[i + 1])
            v_profile[i]  = min(v_profile[i], v_brake_limit)

    # ---- TIME INTEGRATION + LATERAL ACCEL + FUEL ----
    time_profile = np.zeros(n)
    for i in range(n):
        a_lat[i] = v_profile[i] ** 2 / max(radius[i], 1.0)
        if i > 0 and v_profile[i] > 0:
            dt = ds[i] / v_profile[i]
            time_profile[i] = time_profile[i - 1] + dt
            fuel_acum[i] = fuel_acum[i - 1] + (fuel_per_km / 1000.0) * ds[i]

    # ---- TYRE PRESSURE HOT ESTIMATE ----
    # Simplified: p_hot = p_cold + 0.012 bar/degC above 25 degC ambient
    temp_ref = 25.0
    p_tyre_hot = p_tyre_cold + 0.012 * np.maximum(temp_tyre - temp_ref, 0.0)

    return {
        "time_profile": time_profile,
        "v_profile":    v_profile,
        "a_long":       a_long,
        "a_lat":        a_lat,
        "gear_profile": gear_profile,
        "rpm_profile":  rpm_profile,
        "temp_tyre":    temp_tyre,
        "tyre_pressure":p_tyre_hot,
        "fuel_acum":    fuel_acum,
    }


# ---------------------------------------------------------------------------
# STANDING START solver
# ---------------------------------------------------------------------------

def _run_standing_start(
    p: _LegacyVehicleParams,
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    ds: np.ndarray,
    s: np.ndarray,
    radius: np.ndarray,
    mu: float,
    launch_rpm: float,
    wheelspin_limit: float,
    fuel_per_km: float,
    temp_ini: float,
    p_tyre_cold: float,
    torque_map_rpm: list,
    torque_map_nm: list,
) -> dict:
    """
    Standing start: forward pass from v=0 with launch sequence,
    then standard backward pass for braking.

    Launch sequence (first 0.5 s equivalent points):
    - Clutch engagement ramp: traction limited to wheelspin_limit
      slip ratio for the first ~clutch_ramp_dist meters.
    - After launch phase: transitions to normal GGV forward pass.
    """
    g = 9.81
    rho = 1.225
    CLUTCH_RAMP_DIST = 30.0  # metres over which clutch fully engages

    v_profile    = np.zeros(n)
    a_long       = np.zeros(n)
    a_lat        = np.zeros(n)
    gear_profile = np.ones(n, dtype=int)
    rpm_profile  = np.zeros(n)
    temp_tyre    = np.ones(n) * temp_ini
    fuel_acum    = np.zeros(n)

    # Initial conditions
    v_profile[0] = 0.0
    gear_profile[0] = 1
    rpm_profile[0]  = launch_rpm

    launch_dist_accum = 0.0

    # ---- FORWARD PASS ----
    for i in range(1, n):
        v_prev  = v_profile[i - 1]
        gear    = _select_gear_optimal(max(v_prev, 0.5), p)
        gear_profile[i] = gear

        rpm = max(_get_rpm(v_prev, gear, p), launch_rpm if v_prev < 5.0 else 0)
        rpm_profile[i - 1] = rpm

        if torque_map_rpm:
            T_engine = _torque_curve_interp(rpm, torque_map_rpm, torque_map_nm, p.rpm_max)
        else:
            T_engine = _torque_curve(rpm, p)

        ratio_total  = p.gear_ratios[gear - 1] * p.final_drive
        F_traction_e = T_engine * ratio_total / p.r_wheel
        F_drag       = 0.5 * rho * p.Cx * p.A_front * v_prev ** 2
        F_downforce  = 0.5 * rho * abs(p.Cl) * p.A_front * v_prev ** 2
        F_normal     = p.m * g + F_downforce

        # Clutch engagement ramp: limit slip during launch
        if launch_dist_accum < CLUTCH_RAMP_DIST:
            clutch_factor = launch_dist_accum / CLUTCH_RAMP_DIST  # 0 → 1
            slip_limit    = wheelspin_limit * (1.0 - clutch_factor) + 0.05
            F_trac_launch = mu * F_normal * (1.0 - slip_limit)
            F_traction    = min(F_traction_e, F_trac_launch)
        else:
            # Normal friction-circle limit
            a_lat_cur  = v_prev ** 2 / max(radius[i], 1.0)
            F_lat_used = p.m * a_lat_cur
            F_trac_grip = np.sqrt(max((mu * F_normal) ** 2 - F_lat_used ** 2, 0.0))
            F_traction  = min(F_traction_e, F_trac_grip)

        launch_dist_accum += ds[i]

        a = (F_traction - F_drag) / p.m
        a_long[i - 1] = a

        v_lat_max = np.sqrt(mu * g * radius[i])
        if ds[i] > 0:
            v_possible   = np.sqrt(max(0.0, v_prev ** 2 + 2 * a * ds[i]))
            v_profile[i] = min(v_possible, v_lat_max)
        else:
            v_profile[i] = v_prev

        a_total = np.sqrt(a ** 2 + (v_prev ** 2 / max(radius[i], 1.0)) ** 2)
        temp_tyre[i] = temp_tyre[i - 1] + 0.05 * a_total

    # ---- BACKWARD PASS ----
    for i in reversed(range(n - 1)):
        v_next      = v_profile[i + 1]
        a_lat_next  = v_next ** 2 / max(radius[i + 1], 1.0)
        a_decel_max = min(np.sqrt(max(0.0, (mu * g) ** 2 - a_lat_next ** 2)), p.max_decel)
        if ds[i + 1] > 0:
            v_profile[i] = min(v_profile[i], np.sqrt(v_next ** 2 + 2 * a_decel_max * ds[i + 1]))

    # ---- TIME INTEGRATION ----
    time_profile = np.zeros(n)
    for i in range(n):
        a_lat[i] = v_profile[i] ** 2 / max(radius[i], 1.0)
        if i > 0 and v_profile[i] > 0:
            dt = ds[i] / v_profile[i]
            time_profile[i] = time_profile[i - 1] + dt
            fuel_acum[i] = fuel_acum[i - 1] + (fuel_per_km / 1000.0) * ds[i]

    p_tyre_hot = p_tyre_cold + 0.012 * np.maximum(temp_tyre - 25.0, 0.0)

    return {
        "time_profile": time_profile,
        "v_profile":    v_profile,
        "a_long":       a_long,
        "a_lat":        a_lat,
        "gear_profile": gear_profile,
        "rpm_profile":  rpm_profile,
        "temp_tyre":    temp_tyre,
        "tyre_pressure":p_tyre_hot,
        "fuel_acum":    fuel_acum,
    }


# ---------------------------------------------------------------------------
# Public entry point: run_simulation
# ---------------------------------------------------------------------------

def run_simulation(
    config: SimulationConfig,
    vehicle_params: VehicleParams,
    circuit,
    save_csv: bool = True,
    out_path: Optional[str] = None,
) -> SimulationResult:
    """
    Main simulation entry point.

    Applies VehicleSetup to VehicleParams, selects solver based on
    SimulationMode, and returns a SimulationResult with all telemetry
    channels and KPIs.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration (mode, setup, tyre compound, etc.).
    vehicle_params : VehicleParams
        Structured vehicle parameters (will not be mutated).
    circuit : object
        Track object with attributes centerline_x, centerline_y.
    save_csv : bool
        If True and out_path is provided, saves telemetry CSV.
    out_path : str | None
        Output CSV file path.

    Returns
    -------
    SimulationResult
    """
    t0 = _time.perf_counter()
    logger.info(f"[SIM] {config.describe()}")

    # 1. Apply setup to a copy of vehicle params
    params_eff = apply_setup_to_params(vehicle_params, config.setup)
    p = _build_flat_params(params_eff)

    # 2. Torque map (use structured map if available)
    torque_map_rpm = params_eff.engine.torque_curve_rpm
    torque_map_nm  = params_eff.engine.torque_curve_nm

    # 3. Track geometry
    x, y, n, ds, s, radius = _compute_track_geometry(circuit)

    # 4. Simulation config scalars
    mu          = params_eff.tire.friction_coefficient
    fuel_per_km = config.setup.__dict__.get("fuel_per_km", 43.0)  # L/100km → L/km
    temp_ini    = config.track_temperature_c + 5.0  # tyre starts slightly above ambient
    p_tyre_cold = config.setup.tyre_pressure_avg_front
    wheelbase   = params_eff.mass_geometry.wheelbase

    # 5. Initial speed
    if config.is_qualifying():
        # Equilibrium speed at track entry (low-speed approximation)
        v0 = 10.0  # m/s
    elif config.is_flying_lap():
        v0 = config.v_entry_kmh / 3.6
    else:  # STANDING_START
        v0 = 0.0

    # 6. Run appropriate solver
    if config.is_standing_start():
        raw = _run_standing_start(
            p=p, x=x, y=y, n=n, ds=ds, s=s, radius=radius,
            mu=mu, launch_rpm=config.launch_rpm,
            wheelspin_limit=config.wheelspin_limit_slip,
            fuel_per_km=fuel_per_km, temp_ini=temp_ini,
            p_tyre_cold=p_tyre_cold,
            torque_map_rpm=torque_map_rpm,
            torque_map_nm=torque_map_nm,
        )
    else:
        raw = _run_ggv_solver(
            p=p, x=x, y=y, n=n, ds=ds, s=s, radius=radius,
            mu=mu, v0=v0, fuel_per_km=fuel_per_km,
            temp_ini=temp_ini, p_tyre_cold=p_tyre_cold,
            torque_map_rpm=torque_map_rpm,
            torque_map_nm=torque_map_nm,
        )

    lap_time = raw["time_profile"][-1]

    # 7. Derive driver input channels
    v_ms    = raw["v_profile"]
    a_long  = raw["a_long"]
    throttle, brake = _driver_inputs_from_accel(a_long, v_ms * 3.6)
    steering = _steering_from_radius(
        radius, v_ms, wheelbase=wheelbase, steering_ratio=15.0
    )

    # 8. Build result object
    result = SimulationResult(
        lap_time        = lap_time,
        mode            = config.mode,
        setup_name      = config.setup.name,
        distance        = s,
        time            = raw["time_profile"],
        v_kmh           = v_ms * 3.6,
        ax_long_g       = a_long / 9.81,
        ay_lat_g        = raw["a_lat"] / 9.81,
        throttle_pct    = throttle,
        brake_pct       = brake,
        steering_deg    = steering,
        gear            = raw["gear_profile"],
        rpm             = raw["rpm_profile"],
        radius          = radius,
        temp_tyre_c     = raw["temp_tyre"],
        tyre_pressure_bar = raw["tyre_pressure"],
        fuel_used_l     = raw["fuel_acum"],
        _a_long_ms2     = a_long,
        _a_lat_ms2      = raw["a_lat"],
    )

    elapsed = _time.perf_counter() - t0
    logger.info(
        f"[PERFORMANCE] GGV Solver Concluído em {elapsed:.4f}s. "
        f"Tempo de volta: {lap_time:.2f}s | "
        f"T_Pneu final: {result.final_tyre_temp_c:.1f}C"
    )
    result.log_kpis()

    if save_csv and out_path:
        result.save_csv(out_path)

    return result


# ---------------------------------------------------------------------------
# Legacy entry point (backwards compatibility)
# ---------------------------------------------------------------------------

def run_bicycle_model(
    params_dict: dict,
    circuit,
    config: dict,
    save_csv: bool = True,
    out_path: Optional[str] = None,
) -> dict:
    """
    Legacy entry point — preserved for backwards compatibility.

    Wraps run_simulation() by constructing VehicleParams and a default
    QUALIFYING SimulationConfig from the flat params_dict and config dict.
    All behaviour is identical to the original implementation.

    Parameters
    ----------
    params_dict : dict
        Flat vehicle parameter dictionary (legacy format).
    circuit : object
        Track object with centerline_x, centerline_y.
    config : dict
        Legacy config dict (keys: coef_aderencia, consumo, temp_pneu_ini).
    save_csv : bool
        Save telemetry to CSV.
    out_path : str | None
        CSV output path.

    Returns
    -------
    dict
        Legacy result dict with keys: lap_time, distance, v_profile,
        a_long, a_lat, gear, rpm, radius, time, temp_pneu, consumo.
    """
    from src.vehicle.parameters import VehicleParams as StructuredVehicleParams
    from src.simulation.simulation_modes import SimulationConfig, SimulationMode
    from src.vehicle.setup import get_default_setup

    # Build structured VehicleParams from flat dict
    vp = StructuredVehicleParams.from_solver_dict(params_dict)

    # Override mu from legacy config if present
    mu_override = config.get("coef_aderencia")
    if mu_override is not None:
        vp.tire.friction_coefficient = float(mu_override)

    # Build SimulationConfig
    sim_config = SimulationConfig(
        mode=SimulationMode.QUALIFYING,
        setup=get_default_setup(),
        track_temperature_c=config.get("track_temp", 35.0),
        tyre_compound="slick_dry",
        export_driver_inputs=True,
    )
    # Override temp_ini from legacy key
    temp_pneu_ini = config.get("temp_pneu_ini", 65.0)
    sim_config.track_temperature_c = temp_pneu_ini - 5.0  # invert offset

    result = run_simulation(
        config=sim_config,
        vehicle_params=vp,
        circuit=circuit,
        save_csv=save_csv,
        out_path=out_path,
    )

    # Return legacy dict format
    return {
        "lap_time":  result.lap_time,
        "distance":  result.distance,
        "v_profile": result.v_kmh / 3.6,  # back to m/s for legacy consumers
        "a_long":    result._a_long_ms2,
        "a_lat":     result._a_lat_ms2,
        "gear":      result.gear,
        "rpm":       result.rpm,
        "radius":    result.radius,
        "time":      result.time,
        "temp_pneu": result.temp_tyre_c,
        "consumo":   result.fuel_used_l,
    }
