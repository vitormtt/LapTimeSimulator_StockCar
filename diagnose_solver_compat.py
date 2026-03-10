"""
Diagnose solver compatibility for all 3 Porsche GT3 Cup profiles.

Probes build_modular_truck_from_dict() with each vehicle's to_solver_dict()
and reports exactly where/why it fails.

Run from project root:
    python diagnose_solver_compat.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.vehicle.fleet import get_vehicle_by_id
from src.simulation.lap_time_solver import build_modular_truck_from_dict

VEHICLES = ["porsche_991_1", "porsche_991_2", "porsche_992_1"]

print("\n--- diagnose_solver_compat ---\n")

for vid in VEHICLES:
    v = get_vehicle_by_id(vid)
    d = v.to_solver_dict()

    print(f"[{vid}] to_solver_dict keys: {sorted(d.keys())}\n")

    try:
        truck = build_modular_truck_from_dict(d)
        print(f"  [PASS] build_modular_truck_from_dict -> ok")
        print(f"         mass={truck.mass} kg  wheelbase={truck.wheelbase} m")
        print(f"         engine type : {type(truck.engine).__name__}")
        print(f"         tires  type : {type(truck.tires).__name__}")
        print(f"         brakes type : {type(truck.brakes).__name__}")
        print(f"         trans  type : {type(truck.transmission).__name__}")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        print("  --- traceback ---")
        traceback.print_exc()
    print()

print("--- done ---\n")
