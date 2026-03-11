"""Modelo point-mass do veículo SNG01 para simulação de tempo de volta.

Referências:
    - Hakewill, J. (2000). Lap Time Simulation.
    - Toyoshima et al. (2017). Lap Time Simulation Technology. Honda R&D Technical Review.
    - CBA Regulamento Técnico Stock Car Pro Series 2026.

Convenções:
    - Sistema SI (m, kg, N, m/s, m/s²).
    - Downforce positiva = força para baixo no veículo.
    - Aceleração de frenagem retornada como valor positivo (magnitude de desaceleração).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import yaml


G: float = 9.81  # m/s² — aceleração gravitacional padrão


@dataclass(frozen=True)
class SNG01:
    """Modelo parametrizado de desempenho point-mass do SNG01.

    Attributes:
        mass_total_kg: Massa total do conjunto (veículo + piloto + combustível).
        wheelbase_m: Entre-eixos em metros.
        cg_to_front_axle_m: Distância do CG ao eixo dianteiro.
        aero_area_m2: Área frontal de referência para coeficientes aerodinâmicos.
        drag_coeff: Coeficiente de arrasto (Cd).
        lift_coeff: Coeficiente de sustentação (Cl < 0 = downforce).
        max_power_w: Potência máxima disponível nas rodas (W).
        mu_longitudinal: Coeficiente de atrito longitudinal (aceleração/frenagem).
        mu_lateral: Coeficiente de atrito lateral (cornering).
        roll_res_c0: Coeficiente de resistência à rolagem constante (N).
        roll_res_c1: Coeficiente de resistência à rolagem velocidade-dependente (N·s/m).
        air_density: Densidade do ar (kg/m³). Default = 1.225 (ISA, nível do mar).
    """

    mass_total_kg: float
    wheelbase_m: float
    cg_to_front_axle_m: float
    aero_area_m2: float
    drag_coeff: float
    lift_coeff: float
    max_power_w: float
    mu_longitudinal: float
    mu_lateral: float
    roll_res_c0: float
    roll_res_c1: float
    air_density: float = 1.225

    # ------------------------------------------------------------------
    # Construtores
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "SNG01":
        """Instancia o modelo a partir de um arquivo YAML de parâmetros.

        Args:
            yaml_path: Caminho para o arquivo YAML (ex: data/vehicles/sng01_base.yaml).

        Returns:
            Instância imutável de SNG01.

        Raises:
            FileNotFoundError: Se o arquivo YAML não existir.
            KeyError: Se campos obrigatórios estiverem ausentes no YAML.
        """
        path = Path(yaml_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        sim = data["simulation_base"]
        aero = data["aero_base"]
        pt = data["powertrain_base"]
        tyre = data["tyre_base"]
        rr = data["resistance_base"]

        return cls(
            mass_total_kg=sim["mass_total_kg"],
            wheelbase_m=sim["wheelbase_m"],
            cg_to_front_axle_m=sim["cg_to_front_axle_m"],
            aero_area_m2=aero["frontal_area_m2"],
            drag_coeff=aero["drag_coefficient"],
            lift_coeff=aero["lift_coefficient"],
            max_power_w=pt["max_power_kw"] * 1_000.0 * pt["driveline_efficiency"],
            mu_longitudinal=tyre["mu_longitudinal"],
            mu_lateral=tyre["mu_lateral"],
            roll_res_c0=rr["rolling_resistance_c0_n"],
            roll_res_c1=rr["rolling_resistance_c1_n_per_mps"],
            air_density=aero.get("air_density_kg_per_m3", 1.225),
        )

    # ------------------------------------------------------------------
    # Propriedades derivadas
    # ------------------------------------------------------------------

    @property
    def weight_n(self) -> float:
        """Peso estático total (N)."""
        return self.mass_total_kg * G

    @property
    def rear_load_fraction(self) -> float:
        """Fração de carga estática no eixo traseiro (layout front-engine RWD)."""
        return self.cg_to_front_axle_m / self.wheelbase_m

    # ------------------------------------------------------------------
    # Forças aerodinâmicas
    # ------------------------------------------------------------------

    def _dynamic_pressure(self, v_mps: float) -> float:
        """Pressão dinâmica q = 0.5 * rho * A * v² (N, com A embutido)."""
        return 0.5 * self.air_density * self.aero_area_m2 * (v_mps ** 2)

    def drag_force_n(self, v_mps: float) -> float:
        """Força de arrasto aerodinâmico (N)."""
        return self._dynamic_pressure(v_mps) * self.drag_coeff

    def downforce_n(self, v_mps: float) -> float:
        """Força de downforce (N ≥ 0). Cl < 0 gera downforce."""
        return max(0.0, -self._dynamic_pressure(v_mps) * self.lift_coeff)

    # ------------------------------------------------------------------
    # Resistência à rolagem
    # ------------------------------------------------------------------

    def rolling_resistance_n(self, v_mps: float) -> float:
        """Força de resistência à rolagem (modelo linear)."""
        return self.roll_res_c0 + self.roll_res_c1 * v_mps

    # ------------------------------------------------------------------
    # Envelopes de desempenho
    # ------------------------------------------------------------------

    def max_accel_mps2(self, v_mps: float) -> float:
        """Aceleração máxima disponível (m/s²) considerando grip traseiro e potência.

        O modelo considera:
        - Limite de aderência longitudinal no eixo traseiro (RWD);
        - Limite de potência na roda (power-limited);
        - Desconta drag aerodinâmico e resistência à rolagem.

        Args:
            v_mps: Velocidade longitudinal (m/s). Valores < 1 m/s usam limite de grip.

        Returns:
            Aceleração resultante (m/s²), sempre ≥ 0.
        """
        driven_normal_load = (self.weight_n + self.downforce_n(v_mps)) * self.rear_load_fraction
        grip_limit_n = self.mu_longitudinal * driven_normal_load
        power_limit_n = self.max_power_w / max(v_mps, 1.0)

        traction_n = min(grip_limit_n, power_limit_n)
        net_n = traction_n - self.drag_force_n(v_mps) - self.rolling_resistance_n(v_mps)
        return max(0.0, net_n / self.mass_total_kg)

    def max_decel_mps2(self, v_mps: float) -> float:
        """Desaceleração máxima de frenagem (m/s²), retornada como valor positivo.

        Considera frenagem de 4 rodas com limite de grip e contribuição do drag.

        Args:
            v_mps: Velocidade longitudinal (m/s).

        Returns:
            Magnitude da desaceleração (m/s²), sempre ≥ 0.
        """
        total_normal_load = self.weight_n + self.downforce_n(v_mps)
        braking_grip_n = self.mu_longitudinal * total_normal_load
        net_n = braking_grip_n + self.drag_force_n(v_mps) + self.rolling_resistance_n(v_mps)
        return max(0.0, net_n / self.mass_total_kg)

    def max_corner_speed_mps(
        self,
        radius_m: float,
        n_iter: int = 10,
        tol_mps: float = 1e-4,
    ) -> float:
        """Velocidade máxima em curva de raio constante (m/s), via iteração ponto-fixo.

        A downforce aerodinâmica aumenta a carga normal e, portanto, a força
        lateral disponível — exigindo convergência iterativa.

        Args:
            radius_m: Raio de curvatura da curva (m). Deve ser > 0.
            n_iter: Número máximo de iterações.
            tol_mps: Tolerância de convergência em m/s.

        Returns:
            Velocidade de equilíbrio em curva (m/s).

        Raises:
            ValueError: Se radius_m <= 0.
        """
        if radius_m <= 0.0:
            raise ValueError(f"radius_m deve ser positivo, recebido: {radius_m}")

        # Estimativa inicial: sem downforce
        v = math.sqrt(radius_m * self.mu_lateral * G)

        for _ in range(n_iter):
            total_normal = self.weight_n + self.downforce_n(v)
            v_new = math.sqrt(radius_m * self.mu_lateral * total_normal / self.mass_total_kg)
            if abs(v_new - v) < tol_mps:
                return v_new
            v = v_new

        return v

    def gg_diagram(
        self,
        v_mps: float,
        n_points: int = 360,
    ) -> list[tuple[float, float]]:
        """Gera pontos do diagrama G-G (Ay vs Ax) para uma velocidade dada.

        Usa a elipse de atrito combinado (friction circle) para aproximar o
        envelope de acelerações combinadas.

        Args:
            v_mps: Velocidade (m/s).
            n_points: Número de pontos no diagrama (resolução angular).

        Returns:
            Lista de tuplas (ax_mps2, ay_mps2) representando o envelope G-G.
        """
        total_normal = self.weight_n + self.downforce_n(v_mps)
        ay_max = self.mu_lateral * total_normal / self.mass_total_kg
        ax_max_accel = self.max_accel_mps2(v_mps)
        ax_max_brake = self.max_decel_mps2(v_mps)

        points: list[tuple[float, float]] = []
        for i in range(n_points):
            theta = 2.0 * math.pi * i / n_points
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # Elipse com eixos assimétricos (aceleração vs frenagem)
            ax_limit = ax_max_accel if cos_t >= 0 else ax_max_brake
            ax = ax_limit * cos_t
            ay = ay_max * sin_t
            points.append((ax, ay))

        return points
