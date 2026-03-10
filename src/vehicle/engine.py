"""
Módulo de motorização modular.
Contém classes para diferentes tipos de powertrain.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d


class BasePowertrain(ABC):
    """Interface abstrata para sistema de propulsão"""

    def __init__(self, config: dict):
        self.config = config
        self.efficiency = config.get('efficiency', 0.9)

    @abstractmethod
    def get_max_torque(self, rpm: float) -> float:
        pass

    @abstractmethod
    def get_max_power(self, rpm: float) -> float:
        pass

    @abstractmethod
    def get_fuel_consumption(self, power: float, dt: float) -> float:
        pass

    def get_wheel_torque(self, engine_torque: float, gear_ratio: float) -> float:
        return engine_torque * gear_ratio * self.efficiency


class ICEEngine(BasePowertrain):
    """Motor de Combustão Interna (Diesel/Gasolina)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.displacement  = config.get('displacement', 0.0)
        self.max_power_kw  = config['max_power_kw']
        self.max_power_rpm = config.get('max_power_rpm', config.get('rpm_max', 3000))
        self.max_torque_nm = config['max_torque_nm']
        self.max_torque_rpm = config.get(
            'max_torque_rpm', config.get('rpm_max', 1300)
        )
        self.idle_rpm    = config.get('idle_rpm', config.get('rpm_idle', 800))
        self.redline_rpm = config.get('redline_rpm', config.get('rpm_max', 3000))
        self._build_torque_curve()

    def _build_torque_curve(self) -> None:
        """
        Build the torque interpolator.

        Priority:
        1. External curve: if config contains both 'torque_curve_rpm' and
           'torque_curve_nm' (e.g. from VehicleParams.to_solver_dict()),
           use them directly after deduplication + sort.
        2. Synthetic 5-point proxy: built from scalar max_torque / max_power
           RPM values. Points are deduplicated with np.unique before
           interpolation to prevent ValueError on coincident RPM values.
        """
        if (
            'torque_curve_rpm' in self.config
            and 'torque_curve_nm' in self.config
        ):
            # --- Path 1: external curve from VehicleParams ---
            rpm_raw = np.array(self.config['torque_curve_rpm'], dtype=float)
            trq_raw = np.array(self.config['torque_curve_nm'],  dtype=float)

            # Sort by RPM (safety: inputs should already be sorted)
            sort_idx = np.argsort(rpm_raw)
            rpm_raw  = rpm_raw[sort_idx]
            trq_raw  = trq_raw[sort_idx]

            # Deduplicate: keep last occurrence for each unique RPM
            _, unique_idx = np.unique(rpm_raw, return_index=True)
            rpm_pts = rpm_raw[unique_idx]
            trq_pts = trq_raw[unique_idx]
        else:
            # --- Path 2: synthetic 5-point proxy (truck / fallback) ---
            rpm_raw = np.array([
                self.idle_rpm,
                self.max_torque_rpm * 0.7,
                self.max_torque_rpm,
                self.max_power_rpm,
                self.redline_rpm,
            ])
            trq_raw = np.array([
                0.30,   # idle
                0.85,   # rising
                1.00,   # peak torque
                0.95,   # max power
                0.80,   # redline
            ]) * self.max_torque_nm

            # Deduplicate: when max_torque_rpm == max_power_rpm == redline_rpm
            # (common for GT3 configs passed without full curve), np.unique
            # removes duplicates and keeps first occurrence.
            _, unique_idx = np.unique(rpm_raw, return_index=True)
            rpm_pts = rpm_raw[unique_idx]
            trq_pts = trq_raw[unique_idx]

        # Need at least 2 points for linear, 4 for cubic
        kind = 'cubic' if len(rpm_pts) >= 4 else 'linear'

        self.torque_curve = interp1d(
            rpm_pts,
            trq_pts,
            kind=kind,
            bounds_error=False,
            fill_value=(trq_pts[0], trq_pts[-1]),
        )

    def get_max_torque(self, rpm: float) -> float:
        rpm_clipped = np.clip(rpm, self.idle_rpm, self.redline_rpm)
        return float(self.torque_curve(rpm_clipped))

    def get_max_power(self, rpm: float) -> float:
        torque = self.get_max_torque(rpm)
        return (torque * rpm * 2 * np.pi) / (60 * 1000)

    def get_fuel_consumption(self, power_kw: float, dt: float) -> float:
        bsfc = self.config.get('bsfc', 210)   # g/kWh
        fuel_density = 0.85                    # kg/L (diesel)
        fuel_rate_kg = (bsfc * power_kw) / (3600 * 1000)
        fuel_rate_L  = fuel_rate_kg / fuel_density
        return fuel_rate_L * dt


class ElectricMotor(BasePowertrain):
    """Motor elétrico (para validação com dados conhecidos)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_torque_nm  = config['max_torque_nm']
        self.max_power_kw   = config['max_power_kw']
        self.base_speed_rpm = config.get('base_speed_rpm', 5000)
        self.max_speed_rpm  = config.get('max_speed_rpm', 15000)
        # Expose idle_rpm / redline_rpm for interface compatibility
        self.idle_rpm    = config.get('idle_rpm', 0)
        self.redline_rpm = config.get('redline_rpm', self.max_speed_rpm)

    def get_max_torque(self, rpm: float) -> float:
        if rpm < self.base_speed_rpm:
            return self.max_torque_nm
        power_limit_torque = (
            (self.max_power_kw * 1000 * 60) / (2 * np.pi * max(rpm, 1))
        )
        return min(self.max_torque_nm, power_limit_torque)

    def get_max_power(self, rpm: float) -> float:
        torque = self.get_max_torque(rpm)
        power_kw = (torque * rpm * 2 * np.pi) / (60 * 1000)
        return min(power_kw, self.max_power_kw)

    def get_fuel_consumption(self, power_kw: float, dt: float) -> float:
        efficiency = 0.95
        return (power_kw / efficiency) * (dt / 3600)
