"""Testes unitários básicos para o modelo SNG01.

Estratégia:
    - Testa monotonia e limites físicos dos envelopes de desempenho.
    - Usa parâmetros embutidos (sem dependência do YAML) para isolamento.
    - Compatível com pytest >= 7.0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.vehicle.sng01 import SNG01, G


@pytest.fixture()
def sng01() -> SNG01:
    """Instância de referência do SNG01 com parâmetros nominais."""
    return SNG01(
        mass_total_kg=1380.0,
        wheelbase_m=2.820,
        cg_to_front_axle_m=1.350,
        aero_area_m2=2.10,
        drag_coeff=0.42,
        lift_coeff=-0.85,
        max_power_w=430_000.0 * 0.92,
        mu_longitudinal=1.70,
        mu_lateral=1.85,
        roll_res_c0=180.0,
        roll_res_c1=6.0,
    )


class TestPhysicalProperties:
    """Verifica propriedades estáticas derivadas."""

    def test_weight_n(self, sng01: SNG01) -> None:
        assert math.isclose(sng01.weight_n, 1380.0 * G, rel_tol=1e-6)

    def test_rear_load_fraction_range(self, sng01: SNG01) -> None:
        frac = sng01.rear_load_fraction
        assert 0.0 < frac < 1.0

    def test_rear_load_fraction_value(self, sng01: SNG01) -> None:
        expected = 1.350 / 2.820
        assert math.isclose(sng01.rear_load_fraction, expected, rel_tol=1e-6)


class TestAerodynamicForces:
    """Verifica comportamento quadrático com velocidade."""

    def test_drag_zero_at_rest(self, sng01: SNG01) -> None:
        assert sng01.drag_force_n(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_downforce_zero_at_rest(self, sng01: SNG01) -> None:
        assert sng01.downforce_n(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_drag_increases_with_speed(self, sng01: SNG01) -> None:
        assert sng01.drag_force_n(50.0) < sng01.drag_force_n(80.0)

    def test_downforce_increases_with_speed(self, sng01: SNG01) -> None:
        assert sng01.downforce_n(50.0) < sng01.downforce_n(80.0)

    def test_downforce_nonnegative(self, sng01: SNG01) -> None:
        for v in [0.0, 10.0, 50.0, 100.0]:
            assert sng01.downforce_n(v) >= 0.0


class TestAccelerationEnvelope:
    """Verifica envelope de aceleração longitudinal."""

    def test_accel_nonnegative(self, sng01: SNG01) -> None:
        for v in [5.0, 20.0, 50.0, 80.0]:
            assert sng01.max_accel_mps2(v) >= 0.0

    def test_accel_decreases_at_high_speed(self, sng01: SNG01) -> None:
        # Power-limited: aceleração deve cair com o aumento de velocidade.
        assert sng01.max_accel_mps2(30.0) > sng01.max_accel_mps2(70.0)


class TestBrakingEnvelope:
    """Verifica envelope de desaceleração."""

    def test_decel_positive(self, sng01: SNG01) -> None:
        for v in [10.0, 50.0, 100.0]:
            assert sng01.max_decel_mps2(v) > 0.0

    def test_decel_increases_with_speed(self, sng01: SNG01) -> None:
        # Drag contribui para frenagem: desaceleração cresce com velocidade.
        assert sng01.max_decel_mps2(50.0) < sng01.max_decel_mps2(100.0)


class TestCornerSpeed:
    """Verifica cálculo de velocidade máxima em curva."""

    def test_corner_speed_positive(self, sng01: SNG01) -> None:
        assert sng01.max_corner_speed_mps(100.0) > 0.0

    def test_corner_speed_increases_with_radius(self, sng01: SNG01) -> None:
        v_small = sng01.max_corner_speed_mps(50.0)
        v_large = sng01.max_corner_speed_mps(200.0)
        assert v_small < v_large

    def test_corner_speed_invalid_radius(self, sng01: SNG01) -> None:
        with pytest.raises(ValueError):
            sng01.max_corner_speed_mps(0.0)

    def test_corner_speed_converges(self, sng01: SNG01) -> None:
        # Resultado deve ser repetível (convergência estável).
        v1 = sng01.max_corner_speed_mps(150.0, n_iter=5)
        v2 = sng01.max_corner_speed_mps(150.0, n_iter=20)
        assert math.isclose(v1, v2, rel_tol=1e-3)


class TestGGDiagram:
    """Verifica geração do diagrama G-G."""

    def test_gg_returns_correct_count(self, sng01: SNG01) -> None:
        pts = sng01.gg_diagram(60.0, n_points=36)
        assert len(pts) == 36

    def test_gg_values_finite(self, sng01: SNG01) -> None:
        for ax, ay in sng01.gg_diagram(60.0):
            assert math.isfinite(ax)
            assert math.isfinite(ay)
