"""
src/vehicle package

Public API for vehicle parameter and setup modules.
"""
from .parameters import (
    VehicleParams,
    VehicleMassGeometry,
    TireParams,
    AeroParams,
    EngineParams,
    TransmissionParams,
    BrakeParams,
    copa_truck_2dof_default,
    porsche_911_gt3_cup_991,
    validate_vehicle_params,
)
from .setup import (
    VehicleSetup,
    apply_setup_to_params,
    validate_setup,
    get_default_setup,
    get_porsche_cup_soft_setup,
    get_porsche_cup_grip_setup,
    ARB_STIFFNESS_NM_DEG,
    WING_AERO_MAP,
)

__all__ = [
    # Parameters
    'VehicleParams',
    'VehicleMassGeometry',
    'TireParams',
    'AeroParams',
    'EngineParams',
    'TransmissionParams',
    'BrakeParams',
    'copa_truck_2dof_default',
    'porsche_911_gt3_cup_991',
    'validate_vehicle_params',
    # Setup
    'VehicleSetup',
    'apply_setup_to_params',
    'validate_setup',
    'get_default_setup',
    'get_porsche_cup_soft_setup',
    'get_porsche_cup_grip_setup',
    'ARB_STIFFNESS_NM_DEG',
    'WING_AERO_MAP',
]
