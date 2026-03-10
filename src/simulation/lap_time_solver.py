"""
Simulador de Lap Time para Caminhão Copa Truck
Modelo 3-DOF (Longitudinal + Lateral + Rolagem) com Pneu Térmico
"""
from src.vehicle.vehicle_model import BicycleVehicle2DOF
from src.vehicle.tires import ThermalPacejkaTire
from src.vehicle.transmission import Transmission
from src.vehicle.brakes import PneumaticBrake
from src.vehicle.engine import ICEEngine
import numpy as np
import pandas as pd
import logging
import time
import sys
from pathlib import Path
from scipy.signal import savgol_filter

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_modular_truck_from_dict(params_dict: dict) -> BicycleVehicle2DOF:
    """
    Instantiate a BicycleVehicle2DOF from a flat solver-dict.

    Key solver_dict fields (all optional, truck defaults shown):
        P_max            [W]       max engine power
        T_max            [Nm]      max engine torque
        rpm_max          [rpm]     redline
        rpm_idle         [rpm]     idle RPM
        torque_curve_rpm list[float]  full curve RPM points (overrides proxy)
        torque_curve_nm  list[float]  full curve torque points
        r_wheel          [m]       wheel radius
        m                [kg]      total mass
        gear_ratios      list[float]  per-gear ratios
        final_drive      [-]       final drive ratio
        mu               [-]       peak tyre friction coefficient
        pacejka_B/C/D/E           Magic Formula coefficients
        P_cold_bar       [bar]     cold tyre pressure (4.5 truck | 1.8 GT3)
        lf, lr           [m]       CG to front/rear axle
        track_width      [m]       average track width
        k_roll           [Nm/rad]  combined ARB roll stiffness
                                   = arb_front_stiffness + arb_rear_stiffness
                                   default 4/4 setup: 230k+185k = 415k Nm/rad
                                   high_df  5/3 setup: 300k+130k = 430k Nm/rad
        h_cg             [m]       CG height
        gear_min         [-]       minimum gear clamp (1 for car, 4 for truck)
    """
    engine_config = {
        'displacement':   params_dict.get('displacement', 12.0),
        'max_power_kw':   params_dict.get('P_max', 600000) / 1000.0,
        'max_power_rpm':  params_dict.get('rpm_max', 2000),
        'max_torque_nm':  params_dict.get('T_max', 3700),
        'max_torque_rpm': params_dict.get('rpm_max', 1300),
        'rpm_max':        params_dict.get('rpm_max', 2800),
        'idle_rpm':       params_dict.get('rpm_idle', 800),
        'redline_rpm':    params_dict.get('rpm_max', 2800),
    }
    if 'torque_curve_rpm' in params_dict and 'torque_curve_nm' in params_dict:
        engine_config['torque_curve_rpm'] = params_dict['torque_curve_rpm']
        engine_config['torque_curve_nm'] = params_dict['torque_curve_nm']

    engine = ICEEngine(engine_config)

    brakes = PneumaticBrake({
        'wheel_radius_m':       params_dict.get('r_wheel', 0.65),
        'max_brake_torque_nm':  (
            params_dict.get('m', 5000)
            * 9.81
            * params_dict.get('r_wheel', 0.65)
            * 1.5
        ),
        'chamber_area_cm2': 800,
    })

    trans = Transmission({
        'gear_ratios': params_dict.get(
            'gear_ratios',
            [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78]
        ),
        'final_drive': params_dict.get('final_drive', 5.33),
    })

    tires = ThermalPacejkaTire({
        'mu_y':        params_dict.get('mu', 1.1),
        'pacejka_b_y': params_dict.get('pacejka_B', 10.0),
        'pacejka_c_y': params_dict.get('pacejka_C', 1.3),
        'pacejka_d_y': params_dict.get('pacejka_D', 1.1),
        'pacejka_E':   params_dict.get('pacejka_E', -0.5),
        'T_initial_C': 65.0,
        'T_ambient_C': 25.0,
        'P_cold_bar':  params_dict.get('P_cold_bar', 4.5),
    })

    lf = params_dict.get('lf', 2.1)
    lr = params_dict.get('lr', 2.3)

    return BicycleVehicle2DOF(
        mass=params_dict.get('m', 5000.0),
        wheelbase=lf + lr,
        a=lf,
        cg_height=params_dict.get('h_cg', 1.1),
        izz=params_dict.get('m', 5000.0) * (lf ** 2 + lr ** 2) / 2,
        engine_sys=engine,
        brake_sys=brakes,
        trans_sys=trans,
        tire_sys=tires,
        track_width=params_dict.get('track_width', 2.45),
        k_roll=params_dict.get('k_roll', 415000.0),
    )


def run_bicycle_model(params_dict, circuit, config, save_csv=True, out_path=None):
    """
    GGV-based forward-backward solver (3-DOF bicycle model).

    config keys:
        coef_aderencia  [-]  override peak friction (default: truck.tires.mu_y)
        gear_min        [-]  minimum gear clamp
                             Copa Truck: gear_min=4  |  GT3 / car: gear_min=1
    """
    start_time = time.time()
    truck = build_modular_truck_from_dict(params_dict)

    g = 9.81
    rho = 1.225
    mu_aderencia = config.get("coef_aderencia", truck.tires.mu_y)
    gear_min: int = int(config.get("gear_min", params_dict.get("gear_min", 1)))

    Cx = params_dict.get('Cx', 0.85)
    A_front = params_dict.get('A_front', 8.7)
    Cl = params_dict.get('Cl', 0.0)

    x_raw = circuit.centerline_x
    y_raw = circuit.centerline_y
    n = len(x_raw)

    window_size = min(51, n // 4)
    if window_size % 2 == 0:
        window_size += 1
    x = savgol_filter(x_raw, window_length=window_size,
                      polyorder=3) if window_size > 3 else x_raw.copy()
    y = savgol_filter(y_raw, window_length=window_size,
                      polyorder=3) if window_size > 3 else y_raw.copy()

    ds = np.zeros(n)
    ds[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.cumsum(ds)

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # P1: suppress divide-by-zero RuntimeWarning on straight segments
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-9) ** 1.5
        radius = np.where(np.abs(curvature) > 1e-6,
                          1.0 / np.abs(curvature), 1e6)
    radius = np.clip(radius, 10.0, 1e6)

    v_profile = np.zeros(n)
    a_long = np.zeros(n)
    a_lat = np.zeros(n)
    gear_profile = np.ones(n, dtype=int)
    rpm_profile = np.zeros(n)
    consumo_acum = np.zeros(n)
    roll_angle_profile = np.zeros(n)
    fz_outer_profile = np.zeros(n)
    slip_angle_est = np.zeros(n)
    temp_pneu_profile = np.zeros(n)
    pressao_pneu_profile = np.zeros(n)
    grip_mult_profile = np.zeros(n)

    lf = truck.a
    lr = truck.wheelbase - lf
    L = truck.wheelbase
    h = truck.cg_height

    # P2: vectorised v_lat_max (replaces Python for-loop, ~10x faster on long circuits)
    # v_lat = sqrt(mu * m * g / (m/R - 0.5*rho*|Cl|*A*mu))
    # Cl is negative for downforce; -0.5*rho*Cl*A*mu > 0 increases available grip
    denom_vec = (truck.mass / radius) - \
        (0.5 * rho * Cl * A_front * mu_aderencia)
    v_lat_max_profile = np.where(
        denom_vec > 0,
        np.sqrt(np.maximum(0.0, (mu_aderencia * truck.mass * g) /
                np.maximum(denom_vec, 1e-9))),
        250.0 / 3.6,
    )

    num_gears = len(truck.transmission.gear_ratios)
    highest_gear_ratio = truck.transmission.get_total_ratio(num_gears)
    absolute_v_rpm_limit = (
        (truck.engine.redline_rpm * 2 * np.pi * truck.brakes.wheel_radius)
        / (60 * highest_gear_ratio)
    )

    # --- FORWARD PASS ---
    v_profile[0] = min(20.0, v_lat_max_profile[0])

    for i in range(1, n):
        v_prev = v_profile[i - 1]
        a_prev = a_long[i - 2] if i > 1 else 0.0

        gear_current = truck.transmission.select_optimal_gear(
            v_prev, truck.brakes.wheel_radius)
        if gear_current < gear_min:
            gear_current = gear_min
        gear_profile[i] = gear_current

        ratio_total = truck.transmission.get_total_ratio(gear_current)
        rpm = (v_prev / truck.brakes.wheel_radius) * \
            ratio_total * 60 / (2 * np.pi)
        rpm = np.clip(rpm, truck.engine.idle_rpm, truck.engine.redline_rpm)
        rpm_profile[i - 1] = rpm

        truck.vx = max(v_prev, 0.1)
        derivadas = truck.calculate_derivatives(
            throttle=1.0, brake_pedal=0.0, steering_angle=0.0, current_rpm=rpm
        )
        F_traction_engine = derivadas['Fx_total']

        F_drag = 0.5 * rho * Cx * A_front * v_prev ** 2
        F_downforce = 0.5 * rho * Cl * A_front * v_prev ** 2

        Fz_rear_static = truck.mass * g * (lf / L)
        Fz_rear_dynamic = Fz_rear_static + truck.mass * \
            a_prev * (h / L) + F_downforce * 0.5
        max_rear_grip = mu_aderencia * Fz_rear_dynamic
        F_lateral = truck.mass * (v_prev ** 2 / radius[i])

        available_long_grip = (
            np.sqrt(max(0.0, max_rear_grip ** 2 - (F_lateral * 0.5) ** 2))
            if max_rear_grip > F_lateral * 0.5 else 0.0
        )

        a = (min(F_traction_engine, available_long_grip) - F_drag) / truck.mass
        a = min(a, 8.0)
        a_long[i - 1] = a

        if ds[i] > 0:
            v_possible = np.sqrt(max(0.0, v_prev ** 2 + 2 * a * ds[i]))
            v_profile[i] = min(
                v_possible, v_lat_max_profile[i], absolute_v_rpm_limit)
        else:
            v_profile[i] = v_prev

    # --- BACKWARD PASS ---
    v_profile[-1] = min(v_profile[-1], v_lat_max_profile[-1])
    for i in reversed(range(n - 1)):
        v_next = v_profile[i + 1]
        F_lateral_next = truck.mass * (v_next ** 2 / radius[i + 1])
        F_downforce_next = 0.5 * rho * Cl * A_front * v_next ** 2

        a_decel_est = mu_aderencia * g
        Delta_Fz_brake = truck.mass * a_decel_est * (h / L)

        Fz_front_dyn = (truck.mass * g * (lr / L)) + \
            Delta_Fz_brake + F_downforce_next * 0.5
        Fz_rear_dyn = max(0.0, (truck.mass * g * (lf / L)) -
                          Delta_Fz_brake + F_downforce_next * 0.5)

        max_total_grip = mu_aderencia * (Fz_front_dyn + Fz_rear_dyn)
        available_brake_grip = (
            np.sqrt(max(0.0, max_total_grip ** 2 - F_lateral_next ** 2))
            if max_total_grip > F_lateral_next else 0.0
        )

        a_decel_brakes = min(
            available_brake_grip / truck.mass,
            truck.brakes.get_max_deceleration(
                v_next, Fz_front_dyn, Fz_rear_dyn)
        )
        a_drag_next = (0.5 * rho * Cx * A_front * v_next ** 2) / truck.mass
        a_decel_effective = min(a_decel_brakes + a_drag_next, 8.0)

        if ds[i + 1] > 0:
            v_profile[i] = min(
                v_profile[i],
                np.sqrt(v_next ** 2 + 2 * a_decel_effective * ds[i + 1])
            )

    # --- TIME + CONSUMO + THERMAL PASS ---
    time_profile = np.zeros(n)
    time_acc = 0.0

    for i in range(n):
        a_lat[i] = v_profile[i] ** 2 / radius[i]

        roll_data = truck.calculate_roll_transfer(a_lat[i])
        sinal_curva = np.sign(curvature[i]) if curvature[i] != 0 else 1.0
        roll_angle_profile[i] = roll_data['roll_angle_deg'] * sinal_curva

        Fz_roda = (truck.mass * g) / 4.0
        fz_outer_profile[i] = Fz_roda + roll_data['delta_fz_lat']

        Fy_front = (truck.mass * a_lat[i] * lr) / L

        if hasattr(truck.tires, 'C_y'):
            cf_total = truck.tires.B_y * truck.tires.C_y * truck.tires.D_y * Fz_roda
        elif hasattr(truck.tires, 'cornering_stiffness'):
            cf_total = truck.tires.cornering_stiffness
        else:
            cf_total = 100000.0

        slip_angle_rad = Fy_front / (cf_total + 1.0)
        slip_angle_est[i] = np.degrees(slip_angle_rad) * sinal_curva

        if i < n - 1:
            a_long[i] = (
                (v_profile[i + 1] ** 2 - v_profile[i] ** 2)
                / (2 * ds[i + 1] if ds[i + 1] > 0 else 1)
            )

        if i > 0 and v_profile[i] > 0:
            dt = ds[i] / v_profile[i]
            time_acc += dt
            time_profile[i] = time_acc

            truck.tires.update_thermal_state(
                slip_angle_rad, Fy_front, v_profile[i], dt)
            temp_pneu_profile[i] = truck.tires.T_core
            pressao_pneu_profile[i] = truck.tires.current_pressure
            grip_mult_profile[i] = truck.tires.current_grip_mult

            F_drag_inst = 0.5 * rho * Cx * A_front * v_profile[i] ** 2
            potencia_kw = max(0.0, truck.mass *
                              a_long[i] + F_drag_inst) * v_profile[i] / 1000
            if potencia_kw > 0:
                consumo_acum[i] = consumo_acum[i - 1] + \
                    truck.engine.get_fuel_consumption(potencia_kw, dt)
            else:
                consumo_acum[i] = consumo_acum[i - 1]
        elif i == 0:
            temp_pneu_profile[0] = truck.tires.T_core
            pressao_pneu_profile[0] = truck.tires.current_pressure
            grip_mult_profile[0] = truck.tires.current_grip_mult

    lap_time = time_profile[-1]
    elapsed = time.time() - start_time
    logger.info(
        f"GGV Solver Concluído em {elapsed:.4f}s. "
        f"Tempo de volta: {lap_time:.2f}s | T_Pneu final: {temp_pneu_profile[-1]:.1f}C"
    )

    # --- Driver input estimates ---
    throttle_pct = np.clip(a_long / (mu_aderencia * g), 0.0, 1.0) * 100.0
    brake_pct = np.clip(-a_long / (mu_aderencia * g), 0.0, 1.0) * 100.0
    # Ackermann steering angle: delta = L / R  (low-speed kinematic approx.)
    steering_deg = np.degrees(L / radius) * np.sign(curvature)

    result = {
        "lap_time":           lap_time,
        "distance":           s,
        "v_profile":          v_profile,
        "a_long":             a_long,
        "a_lat":              a_lat,
        "gear":               gear_profile,
        "rpm":                rpm_profile,
        "radius":             radius,
        "roll_angle_profile": roll_angle_profile,
        "time":               time_profile,
        "consumo":            consumo_acum,
        "temp_pneu":          temp_pneu_profile,
        "pressao_pneu":       pressao_pneu_profile,
        "grip_mult":          grip_mult_profile,
        "front_slip_angle_deg": slip_angle_est,
        "throttle_pct":       throttle_pct,
        "brake_pct":          brake_pct,
        "steering_deg":       steering_deg,
        "compute_time_s":     elapsed,
    }

    if save_csv and out_path:
        g_val = g
        df = pd.DataFrame({
            "Distance":             s,
            "Time":                 time_profile,
            "Speed":                v_profile * 3.6,
            "Engine_RPM":           rpm_profile,
            "Gear":                 gear_profile,
            "G_Long":               a_long / g_val,
            "G_Lat":                a_lat / g_val,
            "Throttle_Pos":         result['throttle_pct'],
            "Brake_Press":          result['brake_pct'],
            "Steering_Angle_deg":   result['steering_deg'],
            "Roll_Angle_deg":       roll_angle_profile,
            "Fz_Outer_Wheel_N":     fz_outer_profile,
            "Front_Slip_Angle_deg": slip_angle_est,
            "Fuel_Cons_Accum_L":    consumo_acum,
            "Tyre_Temp_C":          temp_pneu_profile,
            "Tyre_Press_bar":       pressao_pneu_profile,
            "Tyre_Grip_Mult":       grip_mult_profile,
            "Corner_Radius_m":      radius,
        })
        df.to_csv(out_path, index=False)

    return result
