"""
Módulo de transmissão modular.
"""
import numpy as np

class Transmission:
    """Sistema de transmissão (manual/automática)"""
    
    def __init__(self, config: dict):
        self.gear_ratios = config['gear_ratios']
        self.final_drive = config['final_drive']
        self.efficiency = config.get('efficiency', 0.93)
        self.current_gear = 1
        
        self.upshift_rpm = config.get('upshift_rpm', 2500)
        self.downshift_rpm = config.get('downshift_rpm', 1200)
        
    def get_total_ratio(self, gear: int = None) -> float:
        if gear is None:
            gear = self.current_gear
        return self.gear_ratios[gear - 1] * self.final_drive
    
    def select_optimal_gear(self, vehicle_speed_ms: float, wheel_radius_m: float, target_rpm: float = None) -> int:
        wheel_rpm = (vehicle_speed_ms * 60) / (2 * np.pi * wheel_radius_m)
        best_gear = 1
        best_rpm_diff = float('inf')
        
        for gear in range(1, len(self.gear_ratios) + 1):
            engine_rpm = wheel_rpm * self.get_total_ratio(gear)
            
            if target_rpm is not None:
                rpm_diff = abs(engine_rpm - target_rpm)
                if rpm_diff < best_rpm_diff:
                    best_rpm_diff = rpm_diff
                    best_gear = gear
        
        return best_gear
    
    def auto_shift(self, engine_rpm: float, throttle: float) -> int:
        if engine_rpm > self.upshift_rpm and throttle > 0.8:
            self.current_gear = min(self.current_gear + 1, len(self.gear_ratios))
        elif engine_rpm < self.downshift_rpm and throttle < 0.3:
            self.current_gear = max(self.current_gear - 1, 1)
        
        return self.current_gear
