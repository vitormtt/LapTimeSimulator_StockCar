"""
Vehicle parameters module for lap time simulation.

This module defines all vehicle parameters required for dynamic simulation,
starting with bicycle model (2DOF) and prepared for expansion to higher-fidelity
models (3DOF, 9DOF, 14DOF).

All parameters use SI units. References to academic models are included in docstrings.

Author: Lap Time Simulator Team
Date: 2024-12-01
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class VehicleMassGeometry:
    """
    Mass and geometry parameters for vehicle dynamics.
    
    Essential for bicycle model (2DOF) and all higher-order models.
    References: Rajamani (2012), Pacejka (2012).
    """
    mass: float  # Total vehicle mass [kg]
    lf: float  # Distance from CG to front axle [m]
    lr: float  # Distance from CG to rear axle [m]
    wheelbase: float  # Total wheelbase lf + lr [m]
    track_width: float  # Average track width [m] (for future roll models)
    cg_height: float  # Center of gravity height above ground [m]
    Iz: float  # Yaw moment of inertia [kg·m²]
    Ix: float = 0.0  # Roll moment of inertia [kg·m²] (for 3DOF+)
    Iy: float = 0.0  # Pitch moment of inertia [kg·m²] (for future models)


@dataclass
class TireParams:
    """
    Tire parameters for lateral and longitudinal force generation.
    
    For 2DOF: linear tire model (Cf, Cr, mu).
    For future: Pacejka Magic Formula coefficients.
    References: Pacejka (2012) - Magic Formula.
    """
    # Linear tire model (2DOF)
    cornering_stiffness_front: float  # Front axle cornering stiffness [N/rad]
    cornering_stiffness_rear: float  # Rear axle cornering stiffness [N/rad]
    friction_coefficient: float  # Peak friction coefficient mu [-]
    
    # Tire geometry
    wheel_radius: float  # Effective rolling radius [m]
    
    # Pacejka coefficients (for future nonlinear models)
    pacejka_B: float = 10.0  # Stiffness factor [-]
    pacejka_C: float = 1.3  # Shape factor [-]
    pacejka_D: float = 1.0  # Peak factor [-]
    pacejka_E: float = 0.97  # Curvature factor [-]
    
    # Thermal model (for future tire temperature simulation)
    thermal_capacity: float = 0.0  # Tire thermal capacity [J/K]
    thermal_conductivity: float = 0.0  # Thermal conductivity [W/(m·K)]


@dataclass
class AeroParams:
    """
    Aerodynamic parameters for drag and downforce.
    
    Drag affects longitudinal dynamics (acceleration/braking).
    Downforce affects normal loads and tire friction circle.
    References: Katz (1995) - Race Car Aerodynamics.
    """
    drag_coefficient: float  # Cd [-]
    frontal_area: float  # Frontal area [m²]
    lift_coefficient: float  # Cl [-] (negative = downforce)
    air_density: float = 1.225  # Air density [kg/m³] at sea level, 15°C


@dataclass
class EngineParams:
    """
    Engine/motor parameters for powertrain simulation.
    
    For 2DOF: simplified max power/torque model.
    For future: full torque curve interpolation, thermal limits.
    References: Guzzella & Sciarretta (2013) - Vehicle Propulsion Systems.
    """
    max_power: float  # Maximum power [W]
    max_torque: float  # Maximum torque [N·m]
    rpm_max: float  # Maximum RPM [rev/min]
    rpm_idle: float  # Idle RPM [rev/min]
    rpm_redline: float = 0.0  # Redline RPM (if different from max) [rev/min]
    
    # Engine map (for future torque curve interpolation)
    torque_curve_rpm: List[float] = field(default_factory=list)  # [rev/min]
    torque_curve_nm: List[float] = field(default_factory=list)  # [N·m]
    
    # Thermal limits (for future engine temperature model)
    max_coolant_temp: float = 110.0  # [°C]
    max_oil_temp: float = 130.0  # [°C]


@dataclass
class TransmissionParams:
    """
    Transmission parameters for gear ratio and shift strategy.
    
    Essential for driver model (gear selection, shift timing).
    References: Gillespie (1992) - Fundamentals of Vehicle Dynamics.
    """
    num_gears: int  # Number of forward gears [-]
    gear_ratios: List[float]  # Gear ratios for each gear [-]
    final_drive_ratio: float  # Final drive (differential) ratio [-]
    
    # Shift strategy (for driver model)
    shift_time: float = 0.3  # Time to complete shift [s]
    upshift_rpm: float = 0.0  # RPM threshold for upshift [rev/min]
    downshift_rpm: float = 0.0  # RPM threshold for downshift [rev/min]
    
    # Efficiency
    transmission_efficiency: float = 0.95  # Drivetrain efficiency [-]


@dataclass
class BrakeParams:
    """
    Brake system parameters for deceleration limits.
    
    Critical for lap time optimization (brake point, brake release).
    References: Limpert (1999) - Brake Design and Safety.
    """
    max_brake_force: float  # Maximum total brake force [N]
    brake_balance: float  # Front brake bias [%] (e.g., 60 = 60% front, 40% rear)
    
    # Brake system characteristics
    max_deceleration: float  # Physical limit [m/s²] (tire + brake combined)
    brake_response_time: float = 0.1  # Brake actuator delay [s]
    
    # ABS parameters (for future active safety models)
    abs_enabled: bool = False
    abs_slip_target: float = 0.15  # Target slip ratio for ABS [-]


@dataclass
class VehicleParams:
    """
    Complete vehicle parameter set for lap time simulation.
    
    Combines all subsystem parameters into a single object.
    Used by solver, driver model, and optimization routines.
    """
    # Core subsystems
    mass_geometry: VehicleMassGeometry
    tire: TireParams
    aero: AeroParams
    engine: EngineParams
    transmission: TransmissionParams
    brake: BrakeParams
    
    # Metadata
    name: str = "Unnamed Vehicle"
    manufacturer: str = ""
    year: int = 0
    category: str = "Truck"  # Truck, Formula, GT, etc.
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization or Streamlit session_state."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VehicleParams':
        """Load from dictionary (e.g., from JSON file or session_state)."""
        return cls(
            mass_geometry=VehicleMassGeometry(**data['mass_geometry']),
            tire=TireParams(**data['tire']),
            aero=AeroParams(**data['aero']),
            engine=EngineParams(**data['engine']),
            transmission=TransmissionParams(**data['transmission']),
            brake=BrakeParams(**data['brake']),
            name=data.get('name', 'Unnamed Vehicle'),
            manufacturer=data.get('manufacturer', ''),
            year=data.get('year', 0),
            category=data.get('category', 'Truck')
        )
    
    def save_to_json(self, filepath: str) -> None:
        """Save vehicle parameters to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'VehicleParams':
        """Load vehicle parameters from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_solver_dict(self) -> Dict:
        """
        Convert to flat dictionary compatible with legacy solver interface.
        
        Maps structured parameters to flat dict expected by run_bicycle_model().
        This allows gradual migration from dict-based to dataclass-based interface.
        """
        return {
            # Mass/geometry
            'm': self.mass_geometry.mass,
            'lf': self.mass_geometry.lf,
            'lr': self.mass_geometry.lr,
            'h_cg': self.mass_geometry.cg_height,
            
            # Tires
            'Cf': self.tire.cornering_stiffness_front,
            'Cr': self.tire.cornering_stiffness_rear,
            'mu': self.tire.friction_coefficient,
            'r_wheel': self.tire.wheel_radius,
            
            # Engine
            'P_max': self.engine.max_power,
            'T_max': self.engine.max_torque,
            'rpm_max': self.engine.rpm_max,
            'rpm_idle': self.engine.rpm_idle,
            
            # Transmission
            'n_gears': self.transmission.num_gears,
            'gear_ratios': self.transmission.gear_ratios,
            'final_drive': self.transmission.final_drive_ratio,
            
            # Brakes
            'max_decel': self.brake.max_deceleration,
            
            # Aero
            'Cx': self.aero.drag_coefficient,
            'A_front': self.aero.frontal_area,
            'Cl': self.aero.lift_coefficient
        }
    
    @classmethod
    def from_solver_dict(cls, data: Dict) -> 'VehicleParams':
        """
        Create VehicleParams from flat solver dictionary.
        
        Allows loading from legacy session_state or JSON format.
        """
        # Calculate wheelbase
        wheelbase = data.get('lf', 2.1) + data.get('lr', 2.3)
        
        return cls(
            mass_geometry=VehicleMassGeometry(
                mass=data.get('m', 5000.0),
                lf=data.get('lf', 2.1),
                lr=data.get('lr', 2.3),
                wheelbase=wheelbase,
                track_width=data.get('track_width', 2.55),
                cg_height=data.get('h_cg', 1.1),
                Iz=data.get('Iz', 15000.0),
                Ix=data.get('Ix', 2000.0),
                Iy=data.get('Iy', 18000.0)
            ),
            tire=TireParams(
                cornering_stiffness_front=data.get('Cf', 120000.0),
                cornering_stiffness_rear=data.get('Cr', 120000.0),
                friction_coefficient=data.get('mu', 1.1),
                wheel_radius=data.get('r_wheel', 0.65)
            ),
            aero=AeroParams(
                drag_coefficient=data.get('Cx', 0.85),
                frontal_area=data.get('A_front', 8.7),
                lift_coefficient=data.get('Cl', 0.0)
            ),
            engine=EngineParams(
                max_power=data.get('P_max', 600000.0),
                max_torque=data.get('T_max', 3700.0),
                rpm_max=data.get('rpm_max', 2800.0),
                rpm_idle=data.get('rpm_idle', 800.0)
            ),
            transmission=TransmissionParams(
                num_gears=data.get('n_gears', 12),
                gear_ratios=data.get('gear_ratios', [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78]),
                final_drive_ratio=data.get('final_drive', 5.33)
            ),
            brake=BrakeParams(
                max_brake_force=data.get('max_brake_force', 50000.0),
                brake_balance=data.get('brake_balance', 58.0),
                max_deceleration=data.get('max_decel', 7.5)
            ),
            name=data.get('name', 'Unnamed Vehicle'),
            manufacturer=data.get('manufacturer', ''),
            year=data.get('year', 0),
            category=data.get('category', 'Truck')
        )


# ============================================================================
# PRESET VEHICLE MODELS
# ============================================================================

def copa_truck_2dof_default() -> VehicleParams:
    """
    Default Copa Truck parameters for bicycle model (2DOF).
    
    Based on typical Brazilian Copa Truck specifications:
    - Mercedes-Benz Actros or similar
    - ~600 kW diesel engine
    - ~5000 kg race weight
    
    References: Copa Truck technical regulations (2024).
    """
    return VehicleParams(
        mass_geometry=VehicleMassGeometry(
            mass=5000.0,
            lf=2.1,
            lr=2.3,
            wheelbase=4.4,
            track_width=2.55,
            cg_height=1.1,
            Iz=15000.0,
            Ix=2000.0,
            Iy=18000.0
        ),
        tire=TireParams(
            cornering_stiffness_front=120000.0,
            cornering_stiffness_rear=120000.0,
            friction_coefficient=1.1,
            wheel_radius=0.65
        ),
        aero=AeroParams(
            drag_coefficient=0.85,
            frontal_area=8.7,
            lift_coefficient=0.0
        ),
        engine=EngineParams(
            max_power=600000.0,
            max_torque=3700.0,
            rpm_max=2800.0,
            rpm_idle=800.0
        ),
        transmission=TransmissionParams(
            num_gears=12,
            gear_ratios=[14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78],
            final_drive_ratio=5.33,
            shift_time=0.15
        ),
        brake=BrakeParams(
            max_brake_force=50000.0,
            brake_balance=58.0,
            max_deceleration=7.5
        ),
        name="Copa Truck Default (2DOF)",
        manufacturer="Mercedes-Benz",
        year=2024,
        category="Truck"
    )


def porsche_911_gt3_cup_991() -> VehicleParams:
    """
    Porsche 911 GT3 Cup (Type 991.1 / 991.2) parameters for bicycle model (2DOF).

    Used as validation vehicle for model modularisation and lap time comparison
    against real Porsche Carrera Cup Brasil telemetry data (Pi Toolbox / MoTeC).

    Specifications sourced from:
    - Porsche Carrera Cup Brasil — Manual Técnico 991 Fase 1 e Fase 2 (2021)
    - Porsche 911 GT3 Cup (Type 991) Vehicle Description (IMSA / Porsche Motorsport)

    Key parameters:
    - Engine: 3.8 L flat-6 (991.1) / 4.0 L flat-6 (991.2), NA
    - Max power: 338 kW (460 cv) @ 7500 rpm
    - Max torque: 440 N·m @ 6250 rpm
    - Redline: 8500 rpm (protection cut at 9000 rpm)
    - Gearbox: 6-speed sequential (APS pneumatic actuation)
    - Wheelbase: 2463 mm
    - Track: front 1545 mm / rear 1530 mm (avg 1537 mm)
    - Kerb weight: ~1200 kg (homologated min), race ~1250 kg with driver
    - Brakes: front 380×32 mm 6-piston / rear 380×30 mm 4-piston
    - Tyres: Michelin slick 245/650-18 front, 305/660-18 rear
    - ARB: 7-position adjustable front and rear
    - Rear wing: 9-position adjustable

    Notes on estimated parameters:
    - lf/lr split estimated from 40/60 front/rear weight distribution typical of 911
    - Iz estimated via regression from similar GT3 vehicles (~1500 kg·m²)
    - Cornering stiffness estimated for Michelin slick at nominal load/pressure
    - Aero (Cd, Cl) at wing position 5/9 (mid-range baseline)
    """
    return VehicleParams(
        mass_geometry=VehicleMassGeometry(
            mass=1250.0,           # race weight incl. driver [kg]
            lf=1.08,               # CG to front axle (est. 44% front bias) [m]
            lr=1.38,               # CG to rear axle [m]
            wheelbase=2.463,       # official wheelbase [m]
            track_width=1.537,     # avg front/rear track [m]
            cg_height=0.46,        # estimated CG height, low-slung GT3 [m]
            Iz=1500.0,             # yaw inertia estimated [kg·m²]
            Ix=320.0,              # roll inertia estimated [kg·m²]
            Iy=1600.0,             # pitch inertia estimated [kg·m²]
        ),
        tire=TireParams(
            # Michelin Pilot Sport GT slick — estimated at nominal conditions
            cornering_stiffness_front=75000.0,   # per axle [N/rad]
            cornering_stiffness_rear=90000.0,    # per axle, wider rear tyre [N/rad]
            friction_coefficient=1.55,           # slick tyre on dry asphalt [-]
            wheel_radius=0.330,                  # 18" wheel, ~330 mm eff. radius [m]
            pacejka_B=11.5,
            pacejka_C=1.35,
            pacejka_D=1.55,
            pacejka_E=0.95,
        ),
        aero=AeroParams(
            drag_coefficient=0.38,    # Cd at wing pos 5 (mid), estimated [-]
            frontal_area=1.98,        # estimated frontal area [m²]
            lift_coefficient=-0.60,   # Cl at wing pos 5 (downforce), estimated [-]
        ),
        engine=EngineParams(
            max_power=338000.0,    # 338 kW (460 cv) [W]
            max_torque=440.0,      # 440 N·m @ 6250 rpm [N·m]
            rpm_max=8500.0,        # operational redline [rev/min]
            rpm_idle=1000.0,       # idle [rev/min]
            rpm_redline=9000.0,    # ECU protection cut [rev/min]
            # Approximate torque curve (flat-6 NA character)
            torque_curve_rpm=[1000, 2000, 3000, 4000, 5000, 6000, 6250, 7000, 7500, 8000, 8500],
            torque_curve_nm= [200,  280,  340,  390,  420,  438,  440,  435,  420,  380,  300],
            max_coolant_temp=110.0,
            max_oil_temp=140.0,
        ),
        transmission=TransmissionParams(
            num_gears=6,
            # Sequential 6-speed — ratios estimated from Porsche GT3 Cup gearbox data
            gear_ratios=[3.091, 2.353, 1.895, 1.526, 1.250, 1.029],
            final_drive_ratio=3.444,   # estimated final drive [−]
            shift_time=0.05,           # APS pneumatic shift ~50 ms [s]
            upshift_rpm=8000.0,        # suggested upshift point [rev/min]
            downshift_rpm=4500.0,      # suggested downshift point [rev/min]
            transmission_efficiency=0.97,
        ),
        brake=BrakeParams(
            # 380×32 mm 6-piston front / 380×30 mm 4-piston rear (Porsche/Brembo)
            max_brake_force=28000.0,   # estimated peak total brake force [N]
            brake_balance=58.0,        # ~58% front (Porsche recommends bias −2.0 to 0) [%]
            max_deceleration=18.0,     # ~1.8g peak decel on slicks [m/s²]
            brake_response_time=0.05,  # ABS/hydraulic response [s]
            abs_enabled=True,
            abs_slip_target=0.10,      # ABS target slip ratio (slick, dry) [-]
        ),
        name="Porsche 911 GT3 Cup 991 (2DOF)",
        manufacturer="Porsche",
        year=2021,
        category="GT3",
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_vehicle_params(params: VehicleParams) -> List[str]:
    """
    Validate vehicle parameters for physical consistency.
    
    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    
    # Mass/geometry checks
    if params.mass_geometry.mass <= 0:
        errors.append("Mass must be positive")
    if params.mass_geometry.lf <= 0 or params.mass_geometry.lr <= 0:
        errors.append("Wheelbase components (lf, lr) must be positive")
    if abs(params.mass_geometry.wheelbase - (params.mass_geometry.lf + params.mass_geometry.lr)) > 0.01:
        errors.append("Wheelbase must equal lf + lr")
    
    # Tire checks
    if params.tire.cornering_stiffness_front <= 0 or params.tire.cornering_stiffness_rear <= 0:
        errors.append("Cornering stiffness must be positive")
    if params.tire.friction_coefficient <= 0:
        errors.append("Friction coefficient must be positive")
    
    # Powertrain checks
    if params.engine.max_power <= 0:
        errors.append("Max power must be positive")
    if len(params.transmission.gear_ratios) != params.transmission.num_gears:
        errors.append(f"Number of gear ratios ({len(params.transmission.gear_ratios)}) must match num_gears ({params.transmission.num_gears})")
    
    # Brake checks
    if params.brake.brake_balance < 0 or params.brake.brake_balance > 100:
        errors.append("Brake balance must be between 0 and 100%")
    
    return errors
