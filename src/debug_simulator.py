# src/debug_simulator.py
import numpy as np
import pandas as pd

def debug_torque_curve():
    """Testa a curva de torque do motor"""
    print("\n" + "="*80)
    print("DEBUG: CURVA DE TORQUE")
    print("="*80)
    
    # Parâmetros do motor
    T_max = 3700.0
    rpm_max = 2800.0
    rpm_idle = 800.0
    rpm_torque_max = 1300.0
    
    # Testa em vários RPMs
    test_rpms = [0, 600, 800, 1000, 1300, 1500, 2000, 2600, 2800, 3000]
    
    print(f"\nParâmetros: T_max={T_max} Nm, RPM_max={rpm_max}, RPM_torque_max={rpm_torque_max}")
    print(f"\n{'RPM':<8} {'Torque (Nm)':<15} {'Estado':<30}")
    print("-" * 55)
    
    for rpm in test_rpms:
        if rpm < rpm_idle:
            torque = 0.0
            state = "Inativo (RPM < idle)"
        elif rpm <= rpm_torque_max:
            # Rampa até torque máximo
            torque = T_max * (rpm - rpm_idle) / (rpm_torque_max - rpm_idle)
            state = "Rampa (até pico)"
        elif rpm <= rpm_max:
            # Decréscimo após pico
            torque = T_max * np.exp(-0.0015 * (rpm - rpm_torque_max)**1.2)
            state = "Decaindo"
        else:
            torque = 0.0
            state = "Limitado RPM (acima máximo)"
        
        print(f"{rpm:<8.0f} {torque:<15.1f} {state:<30}")
    
    print("\n⚠️  VERIFICAR:")
    print("1. Torque em marcha baixa (1000 RPM) deve ser significativo (~80% T_max)")
    print("2. Pico deve estar em ~1300 RPM com valor = T_max")
    print("3. Deve cair após pico, mas manter valores > 50% T_max até 2500 RPM")

def debug_gear_selection():
    """Testa seleção de marcha"""
    print("\n" + "="*80)
    print("DEBUG: SELEÇÃO DE MARCHA")
    print("="*80)
    
    gear_ratios = [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78]
    final_drive = 5.33
    r_wheel = 0.65
    rpm_max = 2800.0
    v_test = 20.0  # m/s (72 km/h)
    
    print(f"\nParâmetros: V_test={v_test*3.6:.1f} km/h, RPM_max={rpm_max}, Final Drive={final_drive}")
    print(f"\n{'Marcha':<8} {'RPM Estimada':<15} {'Status':<30}")
    print("-" * 55)     