"""
Vehicle setup module for lap time simulation.

Defines the VehicleSetup dataclass and mappings from discrete setup
positions (ARB, wing, brake bias) to physical parameters used by the
solver.

All physical constants derived from:
- Porsche Carrera Cup Brasil — Manual Técnico 991 Fase 1 e Fase 2 (2021)
- Milliken & Milliken (1995) — Race Car Vehicle Dynamics, SAE International
- Gillespie (1992) — Fundamentals of Vehicle Dynamics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .parameters import VehicleParams


# ---------------------------------------------------------------------------
# ARB stiffness lookup table  [N·m/deg]
# Position 1 = softest / Position 7 = stiffest
# ---------------------------------------------------------------------------
ARB_STIFFNESS_NM_DEG: Dict[int, Dict[str, float]] = {
    1: {"front": 120.0,  "rear": 100.0},
    2: {"front": 200.0,  "rear": 170.0},
    3: {"front": 310.0,  "rear": 260.0},
    4: {"front": 450.0,  "rear": 380.0},
    5: {"front": 620.0,  "rear": 530.0},
    6: {"front": 830.0,  "rear": 710.0},
    7: {"front": 1080.0, "rear": 930.0},
}

# Wing aero lookup  (Cd_total, Cl_total)
WING_AERO_MAP: Dict[int, Tuple[float, float]] = {
    1:  (0.31, -0.30),
    2:  (0.33, -0.38),
    3:  (0.35, -0.46),
    4:  (0.36, -0.53),
    5:  (0.38, -0.60),
    6:  (0.40, -0.68),
    7:  (0.42, -0.76),
    8:  (0.44, -0.84),
    9:  (0.46, -0.92),
}

TYRE_PRESSURE_NOMINAL_BAR: float = 2.5
TYRE_CS_PRESSURE_GAIN: float = 0.035


@dataclass
class VehicleSetup:
    """
    Discrete and continuous adjustable parameters of a race car setup.

    Attributes
    ----------
    arb_front : int   Front ARB position [1–7].
    arb_rear  : int   Rear ARB position [1–7].
    wing_position : int  Rear wing angle [1–9].
    brake_bias_offset : float  Brake bias adjustment [-2.0, +2.0].
    tyre_pressure_fl/fr/rl/rr : float  Cold tyre pressures [bar].
    name : str  Human-readable label.
    """
    arb_front: int = 4
    arb_rear: int = 4
    wing_position: int = 5
    brake_bias_offset: float = 0.0
    tyre_pressure_fl: float = 2.5
    tyre_pressure_fr: float = 2.5
    tyre_pressure_rl: float = 2.5
    tyre_pressure_rr: float = 2.5
    name: str = "Default Setup"

    @property
    def arb_front_nm_rad(self) -> float:
        """Front ARB stiffness [N·m/rad]."""
        return ARB_STIFFNESS_NM_DEG[self.arb_front]["front"] * (180.0 / 3.14159265)

    @property
    def arb_rear_nm_rad(self) -> float:
        """Rear ARB stiffness [N·m/rad]."""
        return ARB_STIFFNESS_NM_DEG[self.arb_rear]["rear"] * (180.0 / 3.14159265)

    @property
    def tyre_pressure_avg_front(self) -> float:
        """Average front cold tyre pressure [bar]."""
        return (self.tyre_pressure_fl + self.tyre_pressure_fr) / 2.0

    @property
    def tyre_pressure_avg_rear(self) -> float:
        """Average rear cold tyre pressure [bar]."""
        return (self.tyre_pressure_rl + self.tyre_pressure_rr) / 2.0


def get_default_setup(name: str = "Default Setup") -> VehicleSetup:
    """Return neutral mid-range baseline setup."""
    return VehicleSetup(name=name)


def get_porsche_cup_soft_setup() -> VehicleSetup:
    """Low-downforce setup for high-speed tracks."""
    return VehicleSetup(
        arb_front=2, arb_rear=5,
        wing_position=3,
        brake_bias_offset=-0.5,
        tyre_pressure_fl=2.4, tyre_pressure_fr=2.4,
        tyre_pressure_rl=2.6, tyre_pressure_rr=2.6,
        name="Porsche Cup — Low Downforce (High Speed)",
    )


def get_porsche_cup_grip_setup() -> VehicleSetup:
    """High-grip/high-downforce setup for technical tracks."""
    return VehicleSetup(
        arb_front=4, arb_rear=3,
        wing_position=8,
        brake_bias_offset=-1.0,
        tyre_pressure_fl=2.5, tyre_pressure_fr=2.5,
        tyre_pressure_rl=2.5, tyre_pressure_rr=2.5,
        name="Porsche Cup — High Downforce (Technical Track)",
    )


def apply_setup_to_params(
    base_params: VehicleParams,
    setup: VehicleSetup,
) -> VehicleParams:
    """
    Apply a VehicleSetup to VehicleParams, returning a modified copy.

    Modifies AeroParams (Cd, Cl), BrakeParams (brake_balance),
    and TireParams (cornering stiffness via tyre pressure scaling).
    """
    import copy
    params = copy.deepcopy(base_params)

    if setup.wing_position in WING_AERO_MAP:
        cd, cl = WING_AERO_MAP[setup.wing_position]
        params.aero.drag_coefficient = cd
        params.aero.lift_coefficient = cl

    params.brake.brake_balance = max(
        0.0, min(100.0, base_params.brake.brake_balance + setup.brake_bias_offset)
    )

    def _cs_scale(p_avg: float) -> float:
        delta_units = (p_avg - TYRE_PRESSURE_NOMINAL_BAR) / 0.1
        return 1.0 + delta_units * TYRE_CS_PRESSURE_GAIN

    params.tire.cornering_stiffness_front *= _cs_scale(setup.tyre_pressure_avg_front)
    params.tire.cornering_stiffness_rear  *= _cs_scale(setup.tyre_pressure_avg_rear)

    return params


def validate_setup(setup: VehicleSetup) -> List[str]:
    """Validate setup for physical/regulatory consistency. Returns list of errors."""
    errors: List[str] = []
    if not (1 <= setup.arb_front <= 7):
        errors.append(f"arb_front={setup.arb_front} out of range [1, 7]")
    if not (1 <= setup.arb_rear <= 7):
        errors.append(f"arb_rear={setup.arb_rear} out of range [1, 7]")
    if not (1 <= setup.wing_position <= 9):
        errors.append(f"wing_position={setup.wing_position} out of range [1, 9]")
    if not (-2.0 <= setup.brake_bias_offset <= 2.0):
        errors.append(
            f"brake_bias_offset={setup.brake_bias_offset} outside recommended "
            "range [-2.0, 2.0]"
        )
    for corner, p in [
        ("FL", setup.tyre_pressure_fl), ("FR", setup.tyre_pressure_fr),
        ("RL", setup.tyre_pressure_rl), ("RR", setup.tyre_pressure_rr),
    ]:
        if not (1.5 <= p <= 3.5):
            errors.append(f"tyre_pressure_{corner}={p:.2f} bar outside range [1.5, 3.5]")
    return errors
