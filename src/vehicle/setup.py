"""
VehicleSetup module — configurable setup parameters for Porsche GT3 Cup.

Models the 5 adjustable parameters available during a race weekend:
    1. ARB front position (1–7): maps to torsional stiffness [Nm/rad]
    2. ARB rear position  (1–7): maps to torsional stiffness [Nm/rad]
    3. Wing position      (1–9): maps to ΔCd and ΔCl_rear increments
    4. Tyre pressure cold (bar): affects cornering stiffness and friction
    5. Brake bias                : front bias offset (Porsche knob: -2.0 to 0)

ARB stiffness model (linear interpolation):
    - Position 1 (softest): K_front = 50.000 Nm/rad, K_rear = 40.000 Nm/rad
    - Position 7 (stiffest): K_front = 420.000 Nm/rad, K_rear = 340.000 Nm/rad
    Source: Khalil (2018) ARB parametrization + manual ENG210319 position table

Wing aerodynamics model:
    - Position 1 (min drag/downforce): ΔCd = -0.04, ΔCl = +0.30 (less downforce)
    - Position 9 (max drag/downforce): ΔCd = +0.06, ΔCl = -0.40 (more downforce)
    Baseline: position 5 (ΔCd = 0.0, ΔCl = 0.0)

Tyre pressure model (linear approximation, Pacejka 2012 §3.3):
    - Reference pressure: 1.8 bar cold (Michelin Pilot Cup2 GT3)
    - ΔCf per 0.1 bar: +1.5% above reference, -2.0% below reference
    - Δmu per 0.1 bar: -0.3% (overinflation reduces contact patch)

Author: Lap Time Simulator Team
Date: 2026-03-10
References:
    - ENG210319 Manual 991.1/991.2 — Carrera Cup Brasil (2021)
    - Khalil, S. et al. (2018). Fuzzy PID for active ARB. SAE Technical Paper.
    - Pacejka, H.B. (2012). Tyre and Vehicle Dynamics, 3rd ed. Butterworth-Heinemann.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from .parameters import VehicleParams, TireParams, AeroParams, BrakeParams


# ---------------------------------------------------------------------------
# ARB stiffness lookup — 7 positions, front and rear [Nm/rad]
# ---------------------------------------------------------------------------
_ARB_FRONT_STIFFNESS: Tuple[float, ...] = (
    50_000.0,   # pos 1 — softest (sub-oversteer tendency with stiff rear)
    100_000.0,  # pos 2
    160_000.0,  # pos 3
    230_000.0,  # pos 4 — mid-range default
    300_000.0,  # pos 5
    360_000.0,  # pos 6
    420_000.0,  # pos 7 — stiffest
)

_ARB_REAR_STIFFNESS: Tuple[float, ...] = (
    40_000.0,   # pos 1
    80_000.0,   # pos 2
    130_000.0,  # pos 3
    185_000.0,  # pos 4 — mid-range default
    245_000.0,  # pos 5
    295_000.0,  # pos 6
    340_000.0,  # pos 7
)

# ---------------------------------------------------------------------------
# Wing aero increments — 9 positions
# Index 0 = position 1 (min), Index 8 = position 9 (max)
# (ΔCd, ΔCl) — ΔCl negative = more downforce
# ---------------------------------------------------------------------------
_WING_DELTA: Tuple[Tuple[float, float], ...] = (
    (-0.040, +0.30),  # pos 1 — min downforce, min drag
    (-0.030, +0.22),  # pos 2
    (-0.015, +0.12),  # pos 3
    (-0.005, +0.04),  # pos 4
    (0.000,  0.00),  # pos 5 — baseline (default)
    (+0.010, -0.08),  # pos 6
    (+0.025, -0.18),  # pos 7
    (+0.045, -0.30),  # pos 8
    (+0.060, -0.40),  # pos 9 — max downforce, max drag
)

# ---------------------------------------------------------------------------
# Tyre pressure reference values
# ---------------------------------------------------------------------------
_TYRE_PRESSURE_REFERENCE: float = 1.8     # [bar] cold reference
_TYRE_PRESSURE_MIN: float = 1.4           # [bar] minimum safe
_TYRE_PRESSURE_MAX: float = 2.4           # [bar] maximum safe
# [fraction] Δcornering_stiffness / bar
_CS_CHANGE_PER_BAR: float = 0.15
# [fraction] Δmu / bar (over-pressure reduces grip)
_MU_CHANGE_PER_BAR: float = -0.03


@dataclass
class VehicleSetup:
    """
    Adjustable setup configuration for Porsche GT3 Cup vehicles.

    All parameters map directly to the adjustment positions available
    in the 991 and 992 homologation rules (ENG210319).

    Attributes:
        arb_front       : ARB dianteira — posição 1 (mole) a 7 (rígido)
        arb_rear        : ARB traseira  — posição 1 (mole) a 7 (rígido)
        wing_position   : Asa traseira  — posição 1 (min downforce) a 9 (max)
        tyre_pressure   : Pressão fria do pneu [bar]
        brake_bias      : Offset do brake bias knob (Porsche: -2.0 a 0)
        setup_name      : Label descritivo para logging/comparação
    """
    arb_front: int = 4
    arb_rear: int = 4
    wing_position: int = 5
    tyre_pressure: float = 1.8          # [bar] cold
    brake_bias: float = -1.0            # Porsche rec. range: -2.0 to 0
    setup_name: str = "default"

    def __post_init__(self) -> None:
        """Validate all parameter ranges."""
        if not 1 <= self.arb_front <= 7:
            raise ValueError(f"arb_front must be 1–7, got {self.arb_front}")
        if not 1 <= self.arb_rear <= 7:
            raise ValueError(f"arb_rear must be 1–7, got {self.arb_rear}")
        if not 1 <= self.wing_position <= 9:
            raise ValueError(
                f"wing_position must be 1–9, got {self.wing_position}")
        if not _TYRE_PRESSURE_MIN <= self.tyre_pressure <= _TYRE_PRESSURE_MAX:
            raise ValueError(
                f"tyre_pressure must be {_TYRE_PRESSURE_MIN}–{_TYRE_PRESSURE_MAX} bar, "
                f"got {self.tyre_pressure}"
            )
        if not -2.0 <= self.brake_bias <= 0.0:
            raise ValueError(
                f"brake_bias must be -2.0 to 0, got {self.brake_bias}")

    @property
    def arb_front_stiffness(self) -> float:
        """Torsional stiffness of front ARB [Nm/rad]."""
        return _ARB_FRONT_STIFFNESS[self.arb_front - 1]

    @property
    def arb_rear_stiffness(self) -> float:
        """Torsional stiffness of rear ARB [Nm/rad]."""
        return _ARB_REAR_STIFFNESS[self.arb_rear - 1]

    @property
    def wing_delta_cd(self) -> float:
        """Aerodynamic drag increment from wing position (relative to baseline pos. 5)."""
        return _WING_DELTA[self.wing_position - 1][0]

    @property
    def wing_delta_cl(self) -> float:
        """Aerodynamic lift (downforce) increment from wing position."""
        return _WING_DELTA[self.wing_position - 1][1]

    @property
    def understeer_tendency(self) -> str:
        """
        Qualitative handling balance based on ARB ratio.

        Replicates the logic from ENG210319 setup guide:
            D1/T7 → oversteer tendency
            D7/T1 → understeer tendency
        """
        ratio = self.arb_front / self.arb_rear
        if ratio < 0.6:
            return "oversteer"
        elif ratio > 1.6:
            return "understeer"
        else:
            return "neutral"

    def to_dict(self) -> dict:
        """Serialize setup to flat dictionary for logging and optimization."""
        return {
            "setup_name": self.setup_name,
            "arb_front": self.arb_front,
            "arb_rear": self.arb_rear,
            "arb_front_stiffness_nm_rad": self.arb_front_stiffness,
            "arb_rear_stiffness_nm_rad": self.arb_rear_stiffness,
            "wing_position": self.wing_position,
            "wing_delta_cd": self.wing_delta_cd,
            "wing_delta_cl": self.wing_delta_cl,
            "tyre_pressure_bar": self.tyre_pressure,
            "brake_bias": self.brake_bias,
            "handling_balance": self.understeer_tendency,
        }


def apply_setup(base_params: VehicleParams, setup: VehicleSetup) -> VehicleParams:
    """
    Apply a VehicleSetup to a base VehicleParams, returning a new modified instance.

    This function is the critical bridge between setup optimization and simulation:
    each candidate setup generates a new VehicleParams that is fed into the solver.

    Modifications applied:
        - tire.cornering_stiffness_front/rear: scaled by tyre pressure delta
        - tire.friction_coefficient: adjusted by tyre pressure delta
        - aero.drag_coefficient: base + ΔCd from wing
        - aero.lift_coefficient: base + ΔCl from wing
        - brake.brake_balance: adjusted by brake_bias offset
        - (ARB stiffness stored in setup, used by future 3DOF+ roll model)

    Args:
        base_params : VehicleParams — baseline vehicle (from fleet/)
        setup       : VehicleSetup — setup configuration to apply

    Returns:
        VehicleParams — new instance with setup applied (base_params unchanged)
    """
    from dataclasses import replace

    # --- Tyre pressure effect on cornering stiffness and friction ---
    pressure_delta = setup.tyre_pressure - _TYRE_PRESSURE_REFERENCE  # [bar]
    cs_scale = 1.0 + _CS_CHANGE_PER_BAR * pressure_delta
    mu_scale = 1.0 + _MU_CHANGE_PER_BAR * pressure_delta

    new_tire = replace(
        base_params.tire,
        cornering_stiffness_front=base_params.tire.cornering_stiffness_front * cs_scale,
        cornering_stiffness_rear=base_params.tire.cornering_stiffness_rear * cs_scale,
        friction_coefficient=base_params.tire.friction_coefficient * mu_scale,
        cold_pressure_bar=setup.tyre_pressure,
    )

    # --- Aerodynamic effect of wing position ---
    new_aero = replace(
        base_params.aero,
        drag_coefficient=base_params.aero.drag_coefficient + setup.wing_delta_cd,
        lift_coefficient=base_params.aero.lift_coefficient + setup.wing_delta_cl,
    )

    # --- Brake bias: Porsche knob offset maps to front % adjustment ---
    # Knob at -1.0 ≈ baseline 52%. Each unit ≈ ~0.5% front shift.
    bias_delta = (setup.brake_bias - (-1.0)) * 0.5  # % change from nominal
    new_brake = replace(
        base_params.brake,
        brake_balance=np.clip(
            base_params.brake.brake_balance + bias_delta, 45.0, 65.0
        ),
    )

    # --- ARB stiffness → k_roll front/rear ---
    k_roll_f = setup.arb_front_stiffness
    k_roll_r = setup.arb_rear_stiffness
    k_roll_combined = k_roll_f + k_roll_r

    return replace(
        base_params,
        tire=new_tire,
        aero=new_aero,
        brake=new_brake,
        k_roll=k_roll_combined,
        k_roll_front=k_roll_f,
        k_roll_rear=k_roll_r,
    )


def get_default_setup(setup_name: str = "default") -> VehicleSetup:
    """
    Return a neutral mid-range setup (all parameters at mid-position).

    Args:
        setup_name: Label for the setup.

    Returns:
        VehicleSetup with ARB 4/4, wing 5, pressure 1.8 bar, bias -1.0.
    """
    return VehicleSetup(
        arb_front=4,
        arb_rear=4,
        wing_position=5,
        tyre_pressure=1.8,
        brake_bias=-1.0,
        setup_name=setup_name,
    )
