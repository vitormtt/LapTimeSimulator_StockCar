"""
Módulo de frenagem modular.
"""
from abc import ABC, abstractmethod

class BaseBrake(ABC):
    """Interface abstrata para sistema de freios"""
    
    def __init__(self, config: dict):
        self.config = config
        self.max_brake_torque = config['max_brake_torque_nm']
        self.brake_balance = config.get('brake_balance', 0.6)
        
    @abstractmethod
    def get_brake_force(self, brake_pressure: float, wheel_speed: float) -> dict:
        pass
    
    @abstractmethod
    def get_max_deceleration(self, vehicle_speed: float, load_front: float, load_rear: float) -> float:
        pass


class HydraulicBrake(BaseBrake):
    """Freio hidráulico convencional (veículo comum)"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.wheel_radius = config['wheel_radius_m']
        self.pad_friction = config.get('pad_friction', 0.4)
        self.response_time = config.get('response_time', 0.1)
        
    def get_brake_force(self, brake_pressure: float, wheel_speed: float) -> dict:
        total_torque = self.max_brake_torque * brake_pressure
        torque_front = total_torque * self.brake_balance
        torque_rear = total_torque * (1 - self.brake_balance)
        
        force_front = torque_front / self.wheel_radius
        force_rear = torque_rear / self.wheel_radius
        
        return {
            'front': force_front,
            'rear': force_rear,
            'total': force_front + force_rear
        }
    
    def get_max_deceleration(self, vehicle_speed: float, load_front: float, load_rear: float) -> float:
        mu_road = 0.9
        max_force_friction = mu_road * (load_front + load_rear)
        max_force_brake = self.max_brake_torque / self.wheel_radius
        max_force = min(max_force_friction, max_force_brake)
        
        mass = (load_front + load_rear) / 9.81
        return max_force / mass


class PneumaticBrake(BaseBrake):
    """Freio pneumático (caminhões pesados)"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.wheel_radius = config['wheel_radius_m']
        self.chamber_area_cm2 = config.get('chamber_area_cm2', 800)
        self.max_air_pressure_bar = config.get('max_pressure_bar', 8.5)
        self.response_time = config.get('response_time', 0.4)
        self.lag_time = config.get('lag_time', 0.2)
        
    def get_brake_force(self, brake_pressure: float, wheel_speed: float) -> dict:
        air_pressure = self.max_air_pressure_bar * brake_pressure
        actuator_force = air_pressure * 1e5 * (self.chamber_area_cm2 / 1e4)
        
        efficiency = 0.85
        brake_torque = actuator_force * self.wheel_radius * efficiency
        
        torque_front = brake_torque * self.brake_balance
        torque_rear = brake_torque * (1 - self.brake_balance)
        
        force_front = torque_front / self.wheel_radius
        force_rear = torque_rear / self.wheel_radius
        
        return {
            'front': force_front,
            'rear': force_rear,
            'total': force_front + force_rear,
            'response_time': self.response_time,
            'lag_time': self.lag_time
        }
    
    def get_max_deceleration(self, vehicle_speed: float, load_front: float, load_rear: float) -> float:
        mu_road = 0.8
        max_force_friction = mu_road * (load_front + load_rear)
        
        max_actuator_force = (self.max_air_pressure_bar * 1e5 * (self.chamber_area_cm2 / 1e4))
        max_force_brake = (max_actuator_force * self.wheel_radius * 0.85) / self.wheel_radius
        
        max_force = min(max_force_friction, max_force_brake)
        mass = (load_front + load_rear) / 9.81
        return max_force / mass
