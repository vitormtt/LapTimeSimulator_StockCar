"""
Simulador de Lap Time para Caminhão Copa Truck
Modelo Bicicleta 2-DOF (Modular) + Aceleração/Frenagem + Motor + Aerodinâmica

IMPLEMENTAÇÕES AVANÇADAS:
- Pilar 1: Diagrama GGV Aerodinâmico (Limitadores acoplados Fz + Downforce)
- Pilar 3: Consumo por Demanda de Potência (Básico BSFC Map ready)
"""
import numpy as np
import pandas as pd
import logging
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vehicle.engine import ICEEngine
from src.vehicle.brakes import PneumaticBrake
from src.vehicle.transmission import Transmission
from src.vehicle.tires import PacejkaTire
from src.vehicle.vehicle_model import BicycleVehicle2DOF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_modular_truck_from_dict(params_dict: dict) -> BicycleVehicle2DOF:
    """Constrói o caminhão modular a partir do dicionário."""
    engine = ICEEngine({
        'displacement': 12.0,
        'max_power_kw': params_dict.get('P_max', 600000) / 1000.0,
        'max_power_rpm': 2000,
        'max_torque_nm': params_dict.get('T_max', 3700),
        'max_torque_rpm': 1300,
        'rpm_max': params_dict.get('rpm_max', 2800),
        'rpm_idle': params_dict.get('rpm_idle', 800)
    })
    
    brakes = PneumaticBrake({
        'wheel_radius_m': params_dict.get('r_wheel', 0.65),
        'max_brake_torque_nm': params_dict.get('m', 5000) * 9.81 * params_dict.get('r_wheel', 0.65) * 1.5,
        'chamber_area_cm2': 800
    })
    
    trans = Transmission({
        'gear_ratios': params_dict.get('gear_ratios', [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78]),
        'final_drive': params_dict.get('final_drive', 5.33)
    })
    
    tires = PacejkaTire({
        'mu_y': params_dict.get('mu', 1.1),
        'pacejka_b_y': 10.0,
        'pacejka_c_y': 1.3
    })
    
    return BicycleVehicle2DOF(
        mass=params_dict.get('m', 5000.0),
        wheelbase=params_dict.get('lf', 2.1) + params_dict.get('lr', 2.3),
        a=params_dict.get('lf', 2.1),
        cg_height=params_dict.get('h_cg', 1.1),
        izz=params_dict.get('m', 5000.0) * (params_dict.get('lf', 2.1)**2 + params_dict.get('lr', 2.3)**2) / 2,
        engine_sys=engine,
        brake_sys=brakes,
        trans_sys=trans,
        tire_sys=tires
    )

def run_bicycle_model(params_dict, circuit, config, save_csv=True, out_path=None):
    """
    Simulação Quasi-Steady-State baseada na arquitetura modular e GGV Diagram.
    """
    start_time = time.time()
    truck = build_modular_truck_from_dict(params_dict)
    
    g = 9.81
    rho = 1.225
    mu_aderencia = config.get("coef_aderencia", truck.tires.mu_y)
    
    # Parâmetros aerodinâmicos reais
    Cx = params_dict.get('Cx', 0.85)
    A_front = params_dict.get('A_front', 8.7)
    Cl = params_dict.get('Cl', 0.0)  # Negativo = sustentação (Lift), Positivo = Downforce
    
    # Setup de pista
    x = circuit.centerline_x
    y = circuit.centerline_y
    n = len(x)
    ds = np.zeros(n)
    ds[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.cumsum(ds)
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
    radius = np.where(np.abs(curvature) > 1e-6, 1.0 / np.abs(curvature), 1e6)
    radius = np.clip(radius, 10.0, 1e6)
    
    v_profile = np.zeros(n)
    a_long = np.zeros(n)
    a_lat = np.zeros(n)
    gear_profile = np.ones(n, dtype=int)
    rpm_profile = np.zeros(n)
    consumo_acum = np.zeros(n)
    
    # PILAR 1: GGV DIAGRAM (Aerodinâmica Dinâmica)
    v_lat_max_profile = np.zeros(n)
    for i in range(n):
        denominador = (truck.mass / radius[i]) - (0.5 * rho * Cl * A_front * mu_aderencia)
        if denominador > 0:
            v_lat_max_profile[i] = np.sqrt((mu_aderencia * truck.mass * g) / denominador)
        else:
            v_lat_max_profile[i] = 200.0 / 3.6 # Se downforce infinito, limita velocidade terminal
    
    # FORWARD PASS (Aceleração)
    # Lógica de largada: Caminhões de corrida costumam largar na faixa de 60km/h a 80km/h (rolling start) 
    # ou de 4ª/5ª marcha. Aqui vamos iniciar com v0 realista de rolling start (ex: 20 m/s ~ 72 km/h)
    start_speed = 20.0
    v_profile[0] = min(start_speed, v_lat_max_profile[0]) 
    
    # A velocidade máxima teórica do caminhão usando a última marcha real (ex: índice 12)
    num_gears = len(truck.transmission.gear_ratios)
    highest_gear_ratio = truck.transmission.get_total_ratio(num_gears)
    absolute_v_rpm_limit = (truck.engine.redline_rpm * 2 * np.pi * truck.brakes.wheel_radius) / (60 * highest_gear_ratio)
    
    for i in range(1, n):
        v_prev = v_profile[i-1]
        
        # Garante que a marcha escolhida num caminhão de corrida não caia para as de força absurda de arrancada (1-3)
        gear_current = truck.transmission.select_optimal_gear(v_prev, truck.brakes.wheel_radius)
        if gear_current < 4:
            gear_current = 4
        gear_profile[i] = gear_current
        
        ratio_total = truck.transmission.get_total_ratio(gear_current)
        rpm = (v_prev / truck.brakes.wheel_radius) * ratio_total * 60 / (2 * np.pi)
        rpm = np.clip(rpm, truck.engine.idle_rpm, truck.engine.redline_rpm)
        rpm_profile[i-1] = rpm
        
        # Powertrain Physics
        truck.vx = max(v_prev, 0.1)
        derivadas = truck.calculate_derivatives(throttle=1.0, brake_pedal=0.0, steering_angle=0.0, current_rpm=rpm)
        F_traction = derivadas['Fx_total'] 
        
        # Aerodinâmica Dinâmica
        F_drag = 0.5 * rho * Cx * A_front * v_prev**2
        F_downforce = 0.5 * rho * Cl * A_front * v_prev**2
        
        # Elipse de Tração vs Aderência (Círculo de Kamm ajustado para Normal dinámica)
        F_normal_dynamic = (truck.mass * g) + F_downforce
        max_traction_grip = mu_aderencia * F_normal_dynamic
        F_traction_actual = min(F_traction, max_traction_grip)
        
        a = (F_traction_actual - F_drag) / truck.mass
        # Limita aceleração a um valor fisicamente possível para não explodir
        if a > 15.0: a = 15.0
        
        a_long[i-1] = a
        
        if ds[i] > 0:
            v_possible = np.sqrt(max(0, v_prev**2 + 2 * a * ds[i]))
            v_profile[i] = min(v_possible, v_lat_max_profile[i], absolute_v_rpm_limit)
        else:
            v_profile[i] = v_prev
            
    # BACKWARD PASS (Frenagem integrando GGV dinâmico)
    for i in reversed(range(n-1)):
        v_next = v_profile[i+1]
        
        a_lat_next = v_next**2 / max(radius[i+1], 1.0) if v_next > 0 else 0
        F_downforce_next = 0.5 * rho * Cl * A_front * v_next**2
        
        # A aderência total agora se beneficia do downforce na hora de frear
        F_normal_next = (truck.mass * g) + F_downforce_next
        a_total_available = (mu_aderencia * F_normal_next) / truck.mass
        
        # Sobra aderência para frear? (Kamm's Circle / Friction Circle)
        a_decel_max_friction = np.sqrt(max(0, a_total_available**2 - a_lat_next**2))
        
        # Limitado pelo Brake System físico
        a_decel_max_system = truck.brakes.get_max_deceleration(v_next, F_normal_next * 0.6, F_normal_next * 0.4)
        a_decel_max = min(a_decel_max_friction, a_decel_max_system)
        
        # Drag Ajudando a Frear
        a_drag_next = (0.5 * rho * Cx * A_front * v_next**2) / truck.mass
        a_decel_effective = a_decel_max + a_drag_next
        
        if ds[i+1] > 0:
            v_brake_limit = np.sqrt(v_next**2 + 2 * a_decel_effective * ds[i+1])
            v_profile[i] = min(v_profile[i], v_brake_limit)
            
    # TIME E CONSUMO PASS (Pilar 3: Consumo por mapa termodinâmico)
    time_profile = np.zeros(n)
    for i in range(n):
        a_lat[i] = v_profile[i]**2 / max(radius[i], 1.0)
        if i > 0 and v_profile[i] > 0:
            dt = ds[i] / v_profile[i]
            time_profile[i] = time_profile[i-1] + dt
            
            # Re-calcula F_drag para este V exato para achar potencia exigida
            F_drag_inst = 0.5 * rho * Cx * A_front * v_profile[i]**2
            F_traction_req = truck.mass * a_long[i] + F_drag_inst
            
            potencia_kw = (max(0, F_traction_req) * v_profile[i]) / 1000
            if potencia_kw > 0:
                consumo_seg = truck.engine.get_fuel_consumption(potencia_kw, dt)
                consumo_acum[i] = consumo_acum[i-1] + consumo_seg
            else:
                consumo_acum[i] = consumo_acum[i-1]
                
    lap_time = time_profile[-1]
    elapsed = time.time() - start_time
    logger.info(f"GGV Solver Concluído em {elapsed:.4f}s. Tempo de volta: {lap_time:.2f}s")
    
    result = {
        "lap_time": lap_time,
        "distance": s,
        "v_profile": v_profile,
        "a_long": a_long,
        "a_lat": a_lat,
        "gear": gear_profile,
        "rpm": rpm_profile,
        "radius": radius,
        "time": time_profile,
        "consumo": consumo_acum,
        "compute_time_s": elapsed
    }
    
    if save_csv and out_path:
        # Recupera as temps fake ou adiciona algo para não quebrar a interface que espera temp_pneu
        temp_pneu = np.ones(n) * 65.0 
        result["temp_pneu"] = temp_pneu
        
        df = pd.DataFrame({
            "distance_m": s, "x_m": x, "y_m": y, "v_kmh": v_profile * 3.6,
            "a_long_ms2": a_long, "a_lat_ms2": a_lat, "gear": gear_profile,
            "rpm": rpm_profile, "radius_m": radius, "time_s": time_profile,
            "consumo_l": consumo_acum, "temp_pneu_c": temp_pneu
        })
        df.to_csv(out_path, index=False)
        
    return result
