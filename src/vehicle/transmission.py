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
        """Relação total (marcha × final drive)"""
        if gear is None:
            gear = self.current_gear

        # Garante que a marcha passada seja convertida para integer puro,
        # evitando o erro: list indices must be integers or slices, not float
        gear_int = int(gear)

        # Proteção extra para garantir que não procure uma marcha fora do array
        if gear_int < 1:
            gear_int = 1
        elif gear_int > len(self.gear_ratios):
            gear_int = len(self.gear_ratios)

        return self.gear_ratios[gear_int - 1] * self.final_drive

    def select_optimal_gear(self, vehicle_speed_ms: float, wheel_radius_m: float, target_rpm: float = None) -> int:
        """
        Seleciona marcha ótima baseado em velocidade e RPM alvo.

        Uses upshift_rpm / downshift_rpm to define the target RPM band.
        For GT3 Cup (upshift ~8550, downshift ~5700) the band is very
        different from Copa Truck (upshift ~2500, downshift ~1200).
        """
        wheel_rpm = (vehicle_speed_ms * 60) / \
            (2 * np.pi * max(wheel_radius_m, 0.1))
        best_gear = 1
        best_rpm_diff = float('inf')

        rpm_lo = self.downshift_rpm
        rpm_hi = self.upshift_rpm
        rpm_target = (rpm_lo + rpm_hi) / 2.0

        for gear in range(1, len(self.gear_ratios) + 1):
            engine_rpm = wheel_rpm * self.get_total_ratio(gear)

            if target_rpm is not None:
                rpm_diff = abs(engine_rpm - target_rpm)
            else:
                # Pick gear that keeps RPM closest to middle of operating band
                if rpm_lo <= engine_rpm <= rpm_hi:
                    rpm_diff = abs(engine_rpm - rpm_target)
                elif engine_rpm > rpm_hi:
                    # Over-revving — penalise heavily
                    rpm_diff = (engine_rpm - rpm_hi) * 10
                else:
                    # Under-revving — penalise
                    rpm_diff = (rpm_lo - engine_rpm) * 5

            if rpm_diff < best_rpm_diff:
                best_rpm_diff = rpm_diff
                best_gear = gear

        return int(best_gear)

    def auto_shift(self, engine_rpm: float, throttle: float) -> int:
        """Lógica de troca automática"""
        if engine_rpm > self.upshift_rpm and throttle > 0.8:
            self.current_gear = min(
                self.current_gear + 1, len(self.gear_ratios))
        elif engine_rpm < self.downshift_rpm and throttle < 0.3:
            self.current_gear = max(self.current_gear - 1, 1)

        return int(self.current_gear)
