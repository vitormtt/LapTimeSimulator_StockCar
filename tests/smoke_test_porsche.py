"""
Smoke test — Porsche GT3 Cup fleet instantiation, setup application,
and solver-dict compatibility.

Runs without a circuit object: validates only the data layer.
Execute from the project root:

    python -m pytest tests/smoke_test_porsche.py -v
    # or directly:
    python tests/smoke_test_porsche.py

Expected output (all assertions pass):
    [PASS] 991.1 instantiation
    [PASS] 991.2 instantiation
    [PASS] 992.1 instantiation
    [PASS] validate_vehicle_params (all 3 vehicles)
    [PASS] VehicleSetup default
    [PASS] VehicleSetup extremes (pos 1 and pos 7)
    [PASS] apply_setup modifies aero
    [PASS] apply_setup modifies tires
    [PASS] to_solver_dict keys
    [PASS] fleet registry list_vehicles()
    [PASS] SimulationConfig.qualifying()
    [PASS] SimulationConfig.standing_start() forces v0=0
    All smoke tests passed.

Author: Lap Time Simulator Team
Date: 2026-03-10
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when running directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vehicle.fleet import get_vehicle_by_id, list_vehicles
from src.vehicle.setup import VehicleSetup, apply_setup, get_default_setup
from src.vehicle.parameters import validate_vehicle_params
from src.simulation.simulation_modes import SimulationMode, SimulationConfig


# ---------------------------------------------------------------------------
# Required keys expected by build_modular_truck_from_dict() in the solver
# ---------------------------------------------------------------------------
_REQUIRED_SOLVER_KEYS = {
    'm', 'lf', 'lr', 'h_cg', 'track_width', 'k_roll',
    'Cf', 'Cr', 'mu', 'r_wheel',
    'P_max', 'T_max', 'rpm_max', 'rpm_idle',
    'n_gears', 'gear_ratios', 'final_drive',
    'max_decel', 'Cx', 'A_front', 'Cl',
}


def _check(condition: bool, label: str) -> None:
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} {label}")
    assert condition, f"Smoke test failed: {label}"


def test_fleet_instantiation() -> None:
    print("\n--- Fleet instantiation ---")
    vehicle_ids = ["porsche_991_1", "porsche_991_2", "porsche_992_1"]
    vehicles = {}
    for vid in vehicle_ids:
        v = get_vehicle_by_id(vid)
        vehicles[vid] = v
        _check(v is not None, f"{vid} instantiation")
        _check(v.mass_geometry.mass > 0, f"{vid} mass > 0")
        _check(v.engine.max_power > 0, f"{vid} max_power > 0")
        _check(
            abs(v.mass_geometry.wheelbase - (v.mass_geometry.lf + v.mass_geometry.lr)) < 0.01,
            f"{vid} wheelbase = lf + lr"
        )
    return vehicles


def test_validation() -> None:
    print("\n--- Parameter validation ---")
    for vid in ["porsche_991_1", "porsche_991_2", "porsche_992_1"]:
        v = get_vehicle_by_id(vid)
        errors = validate_vehicle_params(v)
        _check(len(errors) == 0, f"validate_vehicle_params {vid} (errors={errors})")


def test_setup() -> None:
    print("\n--- VehicleSetup ---")
    base = get_vehicle_by_id("porsche_991_1")

    # Default setup
    setup_default = get_default_setup("test_default")
    _check(setup_default.arb_front == 4, "VehicleSetup default arb_front=4")
    _check(setup_default.wing_position == 5, "VehicleSetup default wing=5")

    # Extreme setups
    setup_soft = VehicleSetup(arb_front=1, arb_rear=1, wing_position=1,
                               tyre_pressure=1.8, brake_bias=-2.0, setup_name="soft")
    setup_stiff = VehicleSetup(arb_front=7, arb_rear=7, wing_position=9,
                                tyre_pressure=2.0, brake_bias=0.0, setup_name="stiff")
    _check(setup_soft.arb_front_stiffness < setup_stiff.arb_front_stiffness,
           "VehicleSetup: soft ARB < stiff ARB (front)")
    _check(setup_soft.arb_rear_stiffness < setup_stiff.arb_rear_stiffness,
           "VehicleSetup: soft ARB < stiff ARB (rear)")
    _check(setup_soft.wing_delta_cd < setup_stiff.wing_delta_cd,
           "VehicleSetup: wing pos 1 drag < wing pos 9 drag")
    _check(setup_stiff.wing_delta_cl < setup_soft.wing_delta_cl,
           "VehicleSetup: wing pos 9 has more downforce (Cl more negative)")

    # apply_setup modifies aero
    v_soft = apply_setup(base, setup_soft)
    v_stiff = apply_setup(base, setup_stiff)
    _check(v_soft.aero.drag_coefficient < v_stiff.aero.drag_coefficient,
           "apply_setup: wing pos 1 -> lower Cd than pos 9")
    _check(v_stiff.aero.lift_coefficient < v_soft.aero.lift_coefficient,
           "apply_setup: wing pos 9 -> more downforce (Cl more negative)")

    # apply_setup modifies tires (pressure effect)
    setup_high_p = VehicleSetup(tyre_pressure=2.2, setup_name="high_pressure")
    setup_low_p = VehicleSetup(tyre_pressure=1.5, setup_name="low_pressure")
    v_high = apply_setup(base, setup_high_p)
    v_low = apply_setup(base, setup_low_p)
    _check(v_high.tire.cornering_stiffness_front != base.tire.cornering_stiffness_front,
           "apply_setup: pressure changes Cf")
    # Over-pressure reduces friction coefficient
    _check(v_high.tire.friction_coefficient < base.tire.friction_coefficient,
           "apply_setup: high pressure reduces mu")

    # base_params must be unchanged (immutability check)
    _check(base.aero.drag_coefficient == get_vehicle_by_id("porsche_991_1").aero.drag_coefficient,
           "apply_setup: base_params unchanged (immutable)")


def test_solver_dict_keys() -> None:
    print("\n--- Solver dict compatibility ---")
    for vid in ["porsche_991_1", "porsche_991_2", "porsche_992_1"]:
        v = get_vehicle_by_id(vid)
        d = v.to_solver_dict()
        missing = _REQUIRED_SOLVER_KEYS - set(d.keys())
        _check(len(missing) == 0,
               f"{vid} to_solver_dict() has all required keys (missing={missing})")


def test_fleet_registry() -> None:
    print("\n--- Fleet registry ---")
    fleet = list_vehicles()
    _check("porsche_991_1" in fleet, "list_vehicles() contains porsche_991_1")
    _check("porsche_991_2" in fleet, "list_vehicles() contains porsche_991_2")
    _check("porsche_992_1" in fleet, "list_vehicles() contains porsche_992_1")
    _check(len(fleet) == 3, f"list_vehicles() returns 3 vehicles (got {len(fleet)})")


def test_simulation_modes() -> None:
    print("\n--- SimulationConfig ---")
    cfg_q = SimulationConfig.qualifying(track_id="interlagos")
    _check(cfg_q.mode == SimulationMode.QUALIFYING, "qualifying() mode")
    _check(cfg_q.lap_count == 1, "qualifying() lap_count=1")

    cfg_ss = SimulationConfig.standing_start(track_id="interlagos")
    _check(cfg_ss.mode == SimulationMode.STANDING_START, "standing_start() mode")
    _check(cfg_ss.v0 == 0.0, "standing_start() forces v0=0")

    cfg_roll = SimulationConfig.rolling_start(v0_kmh=80.0, track_id="interlagos")
    _check(abs(cfg_roll.v0 - 80.0 / 3.6) < 0.01, "rolling_start() v0 conversion")
    _check(cfg_roll.mode == SimulationMode.ROLLING_START, "rolling_start() mode")


def test_power_ranking() -> None:
    """Sanity check: 992 must have more power than both 991 variants."""
    print("\n--- Power ranking sanity check ---")
    p991_1 = get_vehicle_by_id("porsche_991_1").engine.max_power
    p991_2 = get_vehicle_by_id("porsche_991_2").engine.max_power
    p992_1 = get_vehicle_by_id("porsche_992_1").engine.max_power
    _check(p992_1 > p991_1, f"992 ({p992_1/1000:.0f} kW) > 991.1 ({p991_1/1000:.0f} kW)")
    _check(p992_1 > p991_2, f"992 ({p992_1/1000:.0f} kW) > 991.2 ({p991_2/1000:.0f} kW)")
    _check(p991_1 == p991_2, f"991.1 == 991.2 (equalized class: {p991_1/1000:.0f} kW)")


if __name__ == "__main__":
    test_fleet_instantiation()
    test_validation()
    test_setup()
    test_solver_dict_keys()
    test_fleet_registry()
    test_simulation_modes()
    test_power_ranking()
    print("\n  All smoke tests passed.\n")
