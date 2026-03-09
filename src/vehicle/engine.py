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
        self.displacement = config['displacement']
        self.max_power_kw = config['max_power_kw']
        self.max_power_rpm = config['max_power_rpm']
        self.max_torque_nm = config['max_torque_nm']
        self.max_torque_rpm = config['max_torque_rpm']
        self.idle_rpm = config.get('idle_rpm', 800)
        self.redline_rpm = config.get('redline_rpm', 3000)
        self._build_torque_curve()
        
    def _build_torque_curve(self):
        rpm_points = np.array([
            self.idle_rpm,
            self.max_torque_rpm * 0.7,
            self.max_torque_rpm,
            self.max_power_rpm,
            self.redline_rpm
        ])
        torque_points = np.array([
            0.3,   # Idle
            0.85,  # Subindo
            1.0,   # Pico de torque
            0.95,  # Potência máxima
            0.80   # Redline
        ]) * self.max_torque_nm
        
        self.torque_curve = interp1d(
            rpm_points, 
            torque_points,
            kind='cubic',
            bounds_error=False,
            fill_value=(torque_points[0], torque_points[-1])
        )
    
    def get_max_torque(self, rpm: float) -> float:
        rpm_clipped = np.clip(rpm, self.idle_rpm, self.redline_rpm)
        return float(self.torque_curve(rpm_clipped))
    
    def get_max_power(self, rpm: float) -> float:
        torque = self.get_max_torque(rpm)
        return (torque * rpm * 2 * np.pi) / (60 * 1000)
    
    def get_fuel_consumption(self, power_kw: float, dt: float) -> float:
        bsfc = self.config.get('bsfc', 210)  # g/kWh
        fuel_density = 0.85  # kg/L (diesel)
        fuel_rate_kg = (bsfc * power_kw) / 3600  
        fuel_rate_L = fuel_rate_kg / fuel_density
        return fuel_rate_L * dt


class ElectricMotor(BasePowertrain):
    """Motor elétrico (para validação com dados conhecidos)"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.max_torque_nm = config['max_torque_nm']
        self.max_power_kw = config['max_power_kw']
        self.base_speed_rpm = config.get('base_speed_rpm', 5000)
        self.max_speed_rpm = config.get('max_speed_rpm', 15000)
        
    def get_max_torque(self, rpm: float) -> float:
        if rpm < self.base_speed_rpm:
            return self.max_torque_nm
        else:
            power_limit_torque = (self.max_power_kw * 1000 * 60) / (2 * np.pi * max(rpm, 1))
            return min(self.max_torque_nm, power_limit_torque)
    
    def get_max_power(self, rpm: float) -> float:
        torque = self.get_max_torque(rpm)
        power_kw = (torque * rpm * 2 * np.pi) / (60 * 1000)
        return min(power_kw, self.max_power_kw)
    
    def get_fuel_consumption(self, power_kw: float, dt: float) -> float:
        efficiency = 0.95
        return (power_kw / efficiency) * (dt / 3600)
