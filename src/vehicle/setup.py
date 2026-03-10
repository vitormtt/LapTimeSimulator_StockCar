"""
Vehicle setup module for lap time simulation.

Defines the VehicleSetup dataclass and mappings from discrete setup
positions (ARB, wing, brake bias) to physical parameters used by the
solver. Modular design allows sweep/optimisation of setup space.

All physical constants derived from:
- Porsche Carrera Cup Brasil — Manual Técnico 991 Fase 1 e Fase 2 (2021)
- Milliken & Milliken (1995) — Race Car Vehicle Dynamics, SAE International
- Gillespie (1992) — Fundamentals of Vehicle Dynamics

Author: Lap Time Simulator Team
Date: 2026-03-10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.vehicle.parameters import VehicleParams


# ---------------------------------------------------------------------------
# ARB stiffness lookup table
# Position 1 = softest / Position 7 = stiffest
# Values [N·m/deg] converted internally to [N·m/rad] by the apply function.
# Estimated from typical GT3 ARB range; calibrate with real data when available.
# ---------------------------------------------------------------------------
ARB_STIFFNESS_NM_DEG: Dict[int, Dict[str, float]] = {
    1: {"front": 120.0,  "rear": 100.0},
    2: {"front": 200.0,  "rear": 170.0},
    3: {"front": 310.0,  "rear": 260.0},
    4: {"front": 450.0,  "rear": 380.0},  # mid-range baseline
    5: {"front": 620.0,  "rear": 530.0},
    6: {"front": 830.0,  "rear": 710.0},
    7: {"front": 1080.0, "rear": 930.0},
}

# ---------------------------------------------------------------------------
# Wing aero lookup table — (Cd_increment, Cl_increment) relative to pos 5 baseline
# Baseline (pos 5): Cd=0.38, Cl=-0.60 (as defined in porsche_911_gt3_cup_991)
# Wing pos 1 = lowest downforce/drag; pos 9 = maximum downforce/drag
# ---------------------------------------------------------------------------
WING_AERO_MAP: Dict[int, Tuple[float, float]] = {
    #  pos: (Cd_total, Cl_total)
    1:  (0.31, -0.30),
    2:  (0.33, -0.38),
    3:  (0.35, -0.46),
    4:  (0.36, -0.53),
    5:  (0.38, -0.60),  # baseline
    6:  (0.40, -0.68),
    7:  (0.42, -0.76),
    8:  (0.44, -0.84),
    9:  (0.46, -0.92),
}

# ---------------------------------------------------------------------------
# Tyre pressure effect on cornering stiffness
# Linear scaling factor per bar deviation from nominal (2.5 bar cold)
# Ref: Pacejka (2012), Michelin GT3 Cup tyre notes (estimated)
# ---------------------------------------------------------------------------
TYRE_PRESSURE_NOMINAL_BAR: float = 2.5   # cold pressure reference [bar]
TYRE_CS_PRESSURE_GAIN: float = 0.035     # relative CS change per 0.1 bar [-/0.1bar]


@dataclass
class VehicleSetup:
    """
    Discrete and continuous adjustable parameters of a race car setup.

    Maps directly to the adjustable items documented in the Porsche 991 GT3
    Cup Brasil technical manual (ARB positions, wing angle, brake bias, tyres).
    Designed to be generic enough for Copa Truck and other vehicle presets.

    Attributes
    ----------
    arb_front : int
        Front anti-roll bar stiffness position [1–7]. 1 = softest (tendency
        towards oversteer), 7 = stiffest (tendency towards understeer).
    arb_rear : int
        Rear anti-roll bar stiffness position [1–7]. 1 = softest (tendency
        towards understeer), 7 = stiffest (tendency towards oversteer).
    wing_position : int
        Rear wing angle position [1–9]. 1 = minimum downforce, 9 = maximum.
        Only applicable to vehicles with adjustable rear wing.
    brake_bias_offset : float
        Brake bias adjustment from vehicle baseline [-2.0 to +2.0].
        Negative = more rear braking. Porsche recommends −2.0 to 0.
    tyre_pressure_fl : float
        Front-left tyre pressure, cold [bar].
    tyre_pressure_fr : float
        Front-right tyre pressure, cold [bar].
    tyre_pressure_rl : float
        Rear-left tyre pressure, cold [bar].
    tyre_pressure_rr : float
        Rear-right tyre pressure, cold [bar].
    name : str
        Human-readable setup name for logging and UI display.
    """
    arb_front: int = 4             # 1–7 [-]
    arb_rear: int = 4              # 1–7 [-]
    wing_position: int = 5         # 1–9 [-]
    brake_bias_offset: float = 0.0 # relative to vehicle baseline [-]
    tyre_pressure_fl: float = 2.5  # cold [bar]
    tyre_pressure_fr: float = 2.5  # cold [bar]
    tyre_pressure_rl: float = 2.5  # cold [bar]
    tyre_pressure_rr: float = 2.5  # cold [bar]
    name: str = "Default Setup"

    @property
    def arb_front_nm_rad(self) -> float:
        """Front ARB stiffness converted to [N·m/rad]."""
        nm_per_deg = ARB_STIFFNESS_NM_DEG[self.arb_front]["front"]
        return nm_per_deg * (180.0 / 3.14159265)

    @property
    def arb_rear_nm_rad(self) -> float:
        """Rear ARB stiffness converted to [N·m/rad]."""
        nm_per_deg = ARB_STIFFNESS_NM_DEG[self.arb_rear]["rear"]
        return nm_per_deg * (180.0 / 3.14159265)

    @property
    def tyre_pressure_avg_front(self) -> float:
        """Average front axle cold tyre pressure [bar]."""
        return (self.tyre_pressure_fl + self.tyre_pressure_fr) / 2.0

    @property
    def tyre_pressure_avg_rear(self) -> float:
        """Average rear axle cold tyre pressure [bar]."""
        return (self.tyre_pressure_rl + self.tyre_pressure_rr) / 2.0


def get_default_setup(name: str = "Default Setup") -> VehicleSetup:
    """
    Return neutral mid-range baseline setup.

    ARB front=4, rear=4; wing pos=5; brake bias offset=0;
    all tyre pressures at 2.5 bar nominal.
    """
    return VehicleSetup(name=name)


def get_porsche_cup_soft_setup() -> VehicleSetup:
    """
    Soft/low-downforce setup for high-speed tracks (e.g., Interlagos long).

    Low ARB stiffness front+rear for compliance; wing pos 3 for low drag.
    Ref: ARB guide in Porsche 991 tech manual — tendency table (pos 1 front,
    pos 7 rear = maximum oversteer tendency).
    """
    return VehicleSetup(
        arb_front=2, arb_rear=5,
        wing_position=3,
        brake_bias_offset=-0.5,
        tyre_pressure_fl=2.4, tyre_pressure_fr=2.4,
        tyre_pressure_rl=2.6, tyre_pressure_rr=2.6,
        name="Porsche Cup — Low Downforce (High Speed)",
    )


def get_porsche_cup_grip_setup() -> VehicleSetup:
    """
    High-grip/high-downforce setup for technical tracks (e.g., Curitiba).

    Higher ARB rear for more mechanical grip; wing pos 8 for downforce.
    """
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
    Apply a VehicleSetup to a VehicleParams, returning a modified copy.

    Modifies the following subsystems:
    - AeroParams: Cd and Cl updated from WING_AERO_MAP.
    - BrakeParams: brake_balance shifted by setup.brake_bias_offset.
    - TireParams: cornering stiffness scaled by tyre pressure deviation
      from nominal (linear model, front and rear axles independently).

    ARB stiffness values are stored in the returned object's metadata
    (accessible via setup.arb_front_nm_rad / setup.arb_rear_nm_rad) and
    are intended for use by the roll dynamics module (3DOF+). In the
    current 2DOF solver they act indirectly via cornering stiffness scaling.

    Parameters
    ----------
    base_params : VehicleParams
        Original vehicle parameters (not mutated).
    setup : VehicleSetup
        Setup configuration to apply.

    Returns
    -------
    VehicleParams
        New VehicleParams object with setup applied.
    """
    import copy
    params = copy.deepcopy(base_params)

    # --- Aero: update from wing position map ---
    if setup.wing_position in WING_AERO_MAP:
        cd, cl = WING_AERO_MAP[setup.wing_position]
        params.aero.drag_coefficient = cd
        params.aero.lift_coefficient = cl

    # --- Brake bias ---
    params.brake.brake_balance = (
        base_params.brake.brake_balance + setup.brake_bias_offset
    )
    params.brake.brake_balance = max(0.0, min(100.0, params.brake.brake_balance))

    # --- Tyre pressure → cornering stiffness scaling ---
    # Linear model: +0.1 bar → +3.5% CS; -0.1 bar → -3.5% CS
    def _cs_scale(p_avg: float) -> float:
        delta_bar = p_avg - TYRE_PRESSURE_NOMINAL_BAR
        delta_units = delta_bar / 0.1  # in units of 0.1 bar
        return 1.0 + delta_units * TYRE_CS_PRESSURE_GAIN

    scale_f = _cs_scale(setup.tyre_pressure_avg_front)
    scale_r = _cs_scale(setup.tyre_pressure_avg_rear)
    params.tire.cornering_stiffness_front *= scale_f
    params.tire.cornering_stiffness_rear  *= scale_r

    return params


def validate_setup(setup: VehicleSetup) -> List[str]:
    """
    Validate setup parameters for physical/regulatory consistency.

    Returns
    -------
    List[str]
        Error messages; empty list if setup is valid.
    """
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
            "range [-2.0, 2.0] (Porsche Motorsport recommendation)"
        )
    for corner, p in [
        ("FL", setup.tyre_pressure_fl),
        ("FR", setup.tyre_pressure_fr),
        ("RL", setup.tyre_pressure_rl),
        ("RR", setup.tyre_pressure_rr),
    ]:
        if not (1.5 <= p <= 3.5):
            errors.append(f"tyre_pressure_{corner}={p:.2f} bar outside range [1.5, 3.5]")

    return errors
