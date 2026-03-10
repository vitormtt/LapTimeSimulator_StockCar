"""
Módulo de modelos de Pneus.
Contém modelo Linear (validação rápida) e Pacejka Magic Formula.
Agora inclui modelo Termodinâmico e de Pressão (ThermalPacejkaTire).
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseTire(ABC):
    """Interface abstrata para modelo de pneus"""

    def __init__(self, config: dict):
        self.config = config
        self.mu_x = config.get('mu_x', 0.9)
        self.mu_y = config.get('mu_y', 0.9)

    @abstractmethod
    def get_lateral_force(self, slip_angle: float, normal_load: float) -> float:
        """Calcula força lateral (Fy) com base no slip angle (rad) e carga (N)"""
        pass

    @abstractmethod
    def get_longitudinal_force(self, slip_ratio: float, normal_load: float) -> float:
        """Calcula força longitudinal (Fx) com base no slip ratio e carga (N)"""
        pass


class LinearTire(BaseTire):
    """Modelo Linear - Usado para validação do modelo 2-DOF em baixas acelerações"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.cornering_stiffness = config.get(
            'cornering_stiffness', 80000)  # N/rad
        self.longitudinal_stiffness = config.get(
            'longitudinal_stiffness', 100000)  # N/slip_ratio

    def get_lateral_force(self, slip_angle: float, normal_load: float) -> float:
        fy = -self.cornering_stiffness * slip_angle
        max_fy = self.mu_y * normal_load
        return np.clip(fy, -max_fy, max_fy)

    def get_longitudinal_force(self, slip_ratio: float, normal_load: float) -> float:
        fx = self.longitudinal_stiffness * slip_ratio
        max_fx = self.mu_x * normal_load
        return np.clip(fx, -max_fx, max_fx)


class PacejkaTire(BaseTire):
    """Modelo Magic Formula de Pacejka (Simplificado)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.B_y = config.get('pacejka_b_y', 10.0)
        self.C_y = config.get('pacejka_c_y', 1.3)
        self.D_y = config.get('pacejka_d_y', 1.0)
        self.E_y = config.get('pacejka_e_y', 0.0)

    def get_lateral_force(self, slip_angle: float, normal_load: float) -> float:
        alpha_deg = np.degrees(slip_angle)
        D = self.mu_y * normal_load * self.D_y
        Bx = self.B_y * alpha_deg
        E_term = self.E_y * (Bx - np.arctan(Bx))
        fy = D * np.sin(self.C_y * np.arctan(Bx - E_term))
        return -fy

    def get_longitudinal_force(self, slip_ratio: float, normal_load: float) -> float:
        pass


class ThermalPacejkaTire(PacejkaTire):
    """
    Modelo de Pneu Avançado da Copa Truck.
    Herda o Pacejka, mas adiciona geração de calor por atrito, dissipação convectiva,
    aumento de pressão (Gás Ideal) e degradação de grip por temperatura (Thermal Degradation).
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Estado Inicial
        # Temp da carcaça do pneu (C)
        self.T_core = config.get('T_initial_C', 65.0)
        self.T_ambient = config.get('T_ambient_C', 25.0)
        # Pressão inicial a frio (bar)
        self.P_cold = config.get('P_cold_bar', 1.8)

        # Constantes Térmicas — configurable per vehicle category
        # GT3 Cup tyre: ~12 kg, cp ~1100 J/kg·K, cooling area ~0.5 m²
        # Copa Truck tyre: ~60 kg, cp ~1200 J/kg·K, cooling area ~1.5 m²
        self.mass_tire = config.get('tyre_mass_kg', 12.0)
        self.c_p = config.get('tyre_cp', 1100.0)
        self.h_conv_base = config.get('h_conv_base', 8.0)
        self.area_cooling = config.get('area_cooling', 0.5)

        # Constantes de Grip (Parábola de performance)
        self.T_opt = 95.0      # Temperatura onde o grip é 100% (mu_max)
        self.k_degrad = 0.0003  # Sensibilidade à degradação térmica

        self.current_grip_mult = 1.0
        self.current_pressure = self.P_cold
        self.base_mu_y = self.mu_y

    def update_thermal_state(self, slip_angle_rad: float, fy: float, speed_mps: float, dt: float):
        """
        Integração numérica do estado termodinâmico do pneu a cada passo (dt).
        """
        if speed_mps < 0.1 or dt <= 0:
            return

        # 1. Geração de Calor por Fricção (Friction Power = Força * Velocidade de Deslizamento)
        # V_slip_lat approx V_x * sin(alpha)
        v_slip = speed_mps * np.abs(np.sin(slip_angle_rad))
        power_heat = np.abs(fy) * v_slip  # Watts (J/s)

        # Fator de eficiência (apenas parte da energia fica retida na borracha)
        heat_absorption_factor = 0.35
        heat_in = power_heat * heat_absorption_factor

        # 2. Dissipação de Calor por Convecção (Vento)
        h_conv = self.h_conv_base + (0.5 * speed_mps)
        heat_out = h_conv * self.area_cooling * (self.T_core - self.T_ambient)

        # 3. Equação Diferencial de Temperatura (dT = dQ / (m * c))
        delta_temp = ((heat_in - heat_out) / (self.mass_tire * self.c_p)) * dt
        self.T_core += delta_temp

        # 4. Atualização de Pressão (Lei dos Gases Ideais: P1/T1 = P2/T2)
        # Trabalhando em Kelvin
        T_cold_K = self.T_ambient + 273.15
        T_core_K = self.T_core + 273.15
        self.current_pressure = self.P_cold * (T_core_K / T_cold_K)

        # 5. Atualização do Multiplicador de Grip Térmico (Degradação)
        # Parábola invertida: cai se frio, cai se superaquecido
        grip_mult = 1.0 - self.k_degrad * ((self.T_core - self.T_opt) ** 2)
        # Não perde mais que 25% de grip
        self.current_grip_mult = np.clip(grip_mult, 0.75, 1.0)

        # O mu efetivo muda
        self.mu_y = self.base_mu_y * self.current_grip_mult

    def get_lateral_force(self, slip_angle: float, normal_load: float) -> float:
        # Puxa o calculo padrao da classe pai, mas agora usando o mu_y degradado termicamente
        return super().get_lateral_force(slip_angle, normal_load)
