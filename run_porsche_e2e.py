"""
End-to-end simulation test — Porsche 911 GT3 Cup fleet.

Builds an Interlagos-approximation circuit in memory (no HDF5 file required),
then runs run_bicycle_model() for all 3 Porsche GT3 Cup profiles with two
setup configurations: default (wing 5, ARB 4/4) and high-downforce (wing 9).

Outputs:
    outputs/porsche_e2e/<vehicle_id>_<setup>.csv   — full telemetry channels
    Printed KPI table in the terminal.

Run from project root:
    python run_porsche_e2e.py

Author: Lap Time Simulator Team
Date: 2026-03-10
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.vehicle.fleet import get_vehicle_by_id
from src.vehicle.setup import VehicleSetup, apply_setup
from src.simulation.lap_time_solver import run_bicycle_model
from src.tracks.circuit import CircuitData

# ---------------------------------------------------------------------------
# OUTPUT DIR
# ---------------------------------------------------------------------------
OUT_DIR = Path("outputs/porsche_e2e")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# INTERLAGOS — parametric approximation
# Based on published circuit layout data (Autodromo Jose Carlos Pace)
# Total length: ~4.309 km | 15 turns
# This is a smooth parametric approximation for solver testing only.
# Replace with real GPS centerline for validation.
# ---------------------------------------------------------------------------
def build_interlagos_circuit(n_points: int = 2000) -> CircuitData:
    """
    Build a parametric approximation of Interlagos for solver testing.

    The circuit is constructed from a sequence of straights and circular arcs
    that approximate the key sections: Reta Oposta, S do Senna, Curva 3,
    Descida do Lago, Ferradura, Junco, Mergulho, and Reta Principal.

    Args:
        n_points: Number of centerline points.

    Returns:
        CircuitData with centerline_x, centerline_y and placeholder boundaries.
    """
    # --- Define segments: (type, length_m OR radius_m, angle_deg) ---
    # type: 'straight' -> (length,)  |  'arc' -> (radius, sweep_deg)
    segments = [
        # S do Senna approach
        ("straight", 200),
        ("arc",      55,  -80),   # Curva 1 — left (Senna S)
        ("arc",      45,   70),   # Curva 2 — right
        # Reta Oposta
        ("straight", 570),
        # Curva 3 / Descida do Lago
        ("arc",      80,  -60),
        ("straight", 150),
        ("arc",      50,   45),
        # Ferradura
        ("straight", 200),
        ("arc",      30, -180),   # Ferradura (hairpin)
        # Subida dos Boxes
        ("straight", 380),
        ("arc",      45,  -50),
        # Junco
        ("straight", 150),
        ("arc",      35,   80),
        ("straight", 120),
        # Mergulho
        ("arc",      60,  -90),
        ("straight", 200),
        # Reta Principal
        ("straight", 750),
        ("arc",      90,  -45),   # Curva 1 approach (close loop)
    ]

    # Integrate segments into (x, y, heading) trajectory
    x_pts, y_pts = [0.0], [0.0]
    heading = 0.0  # [degrees] initial heading (pointing along +x)

    for seg in segments:
        if seg[0] == "straight":
            length = seg[1]
            n_seg = max(3, int(length / 2))
            dx = length * np.cos(np.radians(heading))
            dy = length * np.sin(np.radians(heading))
            xs = np.linspace(x_pts[-1], x_pts[-1] + dx, n_seg + 1)[1:]
            ys = np.linspace(y_pts[-1], y_pts[-1] + dy, n_seg + 1)[1:]
            x_pts.extend(xs.tolist())
            y_pts.extend(ys.tolist())

        elif seg[0] == "arc":
            radius, sweep = seg[1], seg[2]  # sweep in degrees
            n_seg = max(4, int(abs(sweep) / 3))
            cx = x_pts[-1] - radius * np.sin(np.radians(heading))
            cy = y_pts[-1] + radius * np.cos(np.radians(heading))
            angles = np.linspace(0, sweep, n_seg + 1)[1:]
            for da in angles:
                ang_rad = np.radians(heading + da - 90)
                x_pts.append(cx + radius * np.cos(ang_rad))
                y_pts.append(cy + radius * np.sin(ang_rad))
            heading += sweep

    x_raw = np.array(x_pts)
    y_raw = np.array(y_pts)

    # Interpolate to n_points
    from scipy.interpolate import interp1d
    s_raw = np.concatenate(([0], np.cumsum(np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2))))
    s_new = np.linspace(0, s_raw[-1], n_points)
    x = interp1d(s_raw, x_raw, kind='linear')(s_new)
    y = interp1d(s_raw, y_raw, kind='linear')(s_new)

    # Simple normal-offset boundaries (constant 10 m half-width)
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-9
    nx = -dy / norm
    ny =  dx / norm
    half_w = 5.5  # [m] half track width

    circuit = CircuitData(
        name="Interlagos (parametric approx.)",
        centerline_x=x,
        centerline_y=y,
        left_boundary_x=x + nx * half_w,
        left_boundary_y=y + ny * half_w,
        right_boundary_x=x - nx * half_w,
        right_boundary_y=y - ny * half_w,
        track_width=np.full(n_points, half_w * 2),
        coordinate_system="local_ENU",
    )
    circuit_length = s_raw[-1]
    print(f"  Circuit: {circuit.name}")
    print(f"  Length : {circuit_length:.0f} m  ({n_points} points)")
    return circuit


# ---------------------------------------------------------------------------
# SIMULATION CONFIG
# ---------------------------------------------------------------------------
SETUPS = {
    "default":       VehicleSetup(arb_front=4, arb_rear=4, wing_position=5,
                                  tyre_pressure=1.8, brake_bias=-1.0,
                                  setup_name="default"),
    "high_downforce": VehicleSetup(arb_front=5, arb_rear=3, wing_position=9,
                                   tyre_pressure=1.85, brake_bias=-0.5,
                                   setup_name="high_downforce"),
}

VEHICLE_IDS = ["porsche_991_1", "porsche_991_2", "porsche_992_1"]

SOLVER_CONFIG = {
    "coef_aderencia": 1.55,   # mu override for circuit conditions (slick)
}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=" * 60)
    print("  Porsche GT3 Cup — E2E Lap Time Simulation")
    print("=" * 60)

    circuit = build_interlagos_circuit(n_points=2000)

    # Header
    header = f"{'Vehicle':<35} {'Setup':<15} {'LapTime':>9} {'Vmax':>8} {'Vmean':>8} {'T_tyre':>8} {'Fuel_L':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    results_all = []

    for vid in VEHICLE_IDS:
        base_params = get_vehicle_by_id(vid)

        for setup_name, setup in SETUPS.items():
            params = apply_setup(base_params, setup)
            solver_dict = params.to_solver_dict()

            # k_roll from actual setup ARB stiffness (combined front+rear)
            solver_dict["k_roll"] = setup.arb_front_stiffness + setup.arb_rear_stiffness

            csv_path = OUT_DIR / f"{vid}_{setup_name}.csv"

            t0 = time.perf_counter()
            try:
                result = run_bicycle_model(
                    params_dict=solver_dict,
                    circuit=circuit,
                    config=SOLVER_CONFIG,
                    save_csv=True,
                    out_path=str(csv_path),
                )
                elapsed = time.perf_counter() - t0

                lap_s   = result["lap_time"]
                v_kmh   = result["v_profile"] * 3.6
                vmax    = float(np.max(v_kmh))
                vmean   = float(np.mean(v_kmh))
                t_tyre  = float(result["temp_pneu"][-1])
                fuel    = float(result["consumo"][-1])

                lap_str = f"{int(lap_s // 60)}:{lap_s % 60:06.3f}"
                print(f"{params.name:<35} {setup_name:<15} {lap_str:>9} "
                      f"{vmax:>7.1f}k {vmean:>7.1f}k {t_tyre:>7.1f}C {fuel:>8.3f}L")
                print(f"  {'':35} {'':15} compute: {elapsed:.3f}s  -> {csv_path.name}")

                results_all.append({
                    "vehicle": vid, "setup": setup_name,
                    "lap_time_s": lap_s, "vmax_kmh": vmax,
                    "vmean_kmh": vmean, "tyre_temp_C": t_tyre,
                    "fuel_L": fuel, "compute_s": elapsed,
                })

            except Exception as exc:
                import traceback
                elapsed = time.perf_counter() - t0
                print(f"{params.name:<35} {setup_name:<15} [FAIL] {type(exc).__name__}: {exc}")
                traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  Done. CSVs saved to: {OUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
