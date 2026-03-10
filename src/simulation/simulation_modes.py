"""
Simulation modes module for lap time simulator.

Defines the SimulationMode enum and SimulationConfig dataclass that
control how the GGV solver is initialised and executed for each
simulation scenario.

Simulation modes
----------------
QUALIFYING
    Single-lap time attack from equilibrium speed at start/finish.
    Target: minimum lap time. Exports full driver input channels
    (throttle, brake, steering, gear, ax, ay) for driver analysis
    and optimisation. This is the primary mode of the simulator.

FLYING_LAP
    Lap started at a specified constant entry speed (v_entry).
    Used for back-to-back setup comparisons under identical
    initial conditions. Also suitable for warm-up and pace-lap
    studies.

STANDING_START
    Lap started from rest (v=0) with launch control sequence.
    Models clutch engagement, wheel-spin, and launch RPM.
    Useful for race-start strategy and traction analysis.

Driver input channels (all modes)
----------------------------------
All modes export the following channels, aligned with Pi Toolbox and
MoTeC telemetry naming conventions (ref: Porsche Carrera Cup Brasil
analysis templates, Pi Toolbox Apostila 2014):

    throttle_pct    : throttle position [0–100 %]
    brake_pct       : brake pedal position [0–100 %]
    steering_deg    : steering wheel angle [deg]
    gear            : engaged gear [1–6 or 1–12]
    ax_long_g       : longitudinal acceleration [g]
    ay_lat_g        : lateral acceleration [g]
    v_kmh           : vehicle speed [km/h]
    rpm             : engine RPM [rev/min]
    distance_m      : cumulative distance along track [m]
    lap_time_s      : cumulative lap time [s]

References
----------
- Segers, J. (2014). Analysis Techniques for Racecar Data Acquisition,
  2nd Ed. SAE International.
- Brayshaw, D.L. & Harrison, M.F. (2005). A quasi steady state approach
  to race car lap simulation. Proc. IMechE, Part D.
- Pi Toolbox Apostila de Treinamento — Porsche Carrera Cup Brasil (2014).

Author: Lap Time Simulator Team
Date: 2026-03-10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.vehicle.setup import VehicleSetup, get_default_setup


class SimulationMode(Enum):
    """
    Enumeration of available simulation scenarios.

    Attributes
    ----------
    QUALIFYING
        Single fastest lap from equilibrium speed. Primary use-case.
    FLYING_LAP
        Lap from a prescribed constant entry speed (v_entry_kmh).
    STANDING_START
        Lap from standstill with launch sequence.
    """
    QUALIFYING    = auto()
    FLYING_LAP    = auto()
    STANDING_START = auto()


@dataclass
class SimulationConfig:
    """
    Full configuration for a single simulation run.

    Bundles SimulationMode, VehicleSetup, and per-mode parameters into
    one object passed to the solver entry point.

    Parameters
    ----------
    mode : SimulationMode
        Simulation scenario to execute.
    setup : VehicleSetup
        Car setup (ARB, wing, brake bias, tyre pressures).
    n_laps : int
        Number of laps to simulate. For QUALIFYING and FLYING_LAP
        typically 1; for STANDING_START use 2+ to include out-lap.
    v_entry_kmh : float
        Initial vehicle speed for FLYING_LAP mode [km/h].
        Ignored in QUALIFYING (speed computed from track equilibrium)
        and STANDING_START (v=0).
    launch_rpm : float
        Engine RPM at clutch release for STANDING_START mode [rev/min].
        Ignored in QUALIFYING and FLYING_LAP.
    track_temperature_c : float
        Ambient track surface temperature [°C]. Affects tyre model
        warm-up rate and friction coefficient scaling.
    tyre_compound : str
        Tyre compound identifier (e.g., 'slick_dry', 'slick_wet',
        'rain'). Used by the thermal tyre model for mu scaling.
    export_driver_inputs : bool
        If True, solver exports detailed driver input channels
        (throttle, brake, steering, gear, ax, ay) to results CSV.
    notes : str
        Optional free-text annotation stored with simulation results.
    """
    mode: SimulationMode = SimulationMode.QUALIFYING
    setup: VehicleSetup = field(default_factory=get_default_setup)

    # General parameters
    n_laps: int = 1
    track_temperature_c: float = 35.0    # typical Brazilian circuit [degC]
    tyre_compound: str = "slick_dry"
    export_driver_inputs: bool = True
    notes: str = ""

    # FLYING_LAP specific
    v_entry_kmh: float = 100.0

    # STANDING_START specific
    launch_rpm: float = 4500.0           # clutch-drop RPM [rev/min]
    wheelspin_limit_slip: float = 0.25   # max allowed slip ratio at launch [-]

    def is_qualifying(self) -> bool:
        """True if mode is QUALIFYING."""
        return self.mode == SimulationMode.QUALIFYING

    def is_flying_lap(self) -> bool:
        """True if mode is FLYING_LAP."""
        return self.mode == SimulationMode.FLYING_LAP

    def is_standing_start(self) -> bool:
        """True if mode is STANDING_START."""
        return self.mode == SimulationMode.STANDING_START

    def describe(self) -> str:
        """Human-readable summary string for UI and logging."""
        base = (
            f"[{self.mode.name}] Setup='{self.setup.name}' "
            f"Tyres={self.tyre_compound} T_track={self.track_temperature_c}°C"
        )
        if self.is_flying_lap():
            base += f" v_entry={self.v_entry_kmh:.1f} km/h"
        if self.is_standing_start():
            base += f" launch_rpm={self.launch_rpm:.0f} rpm"
        return base


def get_default_config(
    mode: SimulationMode = SimulationMode.QUALIFYING,
    setup: Optional[VehicleSetup] = None,
) -> SimulationConfig:
    """
    Return a ready-to-use SimulationConfig with sensible defaults.

    Parameters
    ----------
    mode : SimulationMode
        Desired simulation mode. Defaults to QUALIFYING.
    setup : VehicleSetup | None
        If None, uses get_default_setup() with neutral mid-range values.

    Returns
    -------
    SimulationConfig
    """
    return SimulationConfig(
        mode=mode,
        setup=setup if setup is not None else get_default_setup(),
    )


# ---------------------------------------------------------------------------
# Driver input channel specification (for documentation and CSV header gen)
# ---------------------------------------------------------------------------

DRIVER_INPUT_CHANNELS = [
    # (channel_name, unit, description)
    ("distance_m",   "m",    "Cumulative distance along track centreline"),
    ("lap_time_s",   "s",    "Cumulative lap time"),
    ("v_kmh",        "km/h", "Vehicle speed"),
    ("ax_long_g",    "g",    "Longitudinal acceleration (+ = accel, - = braking)"),
    ("ay_lat_g",     "g",    "Lateral acceleration (+ = left, - = right)"),
    ("throttle_pct", "%",    "Throttle pedal / drive torque request [0-100]"),
    ("brake_pct",    "%",    "Brake pedal pressure request [0-100]"),
    ("steering_deg", "deg",  "Steering wheel angle (+ = left)"),
    ("gear",         "-",    "Engaged gear number"),
    ("rpm",          "rpm",  "Engine rotational speed"),
]

DRIVER_INPUT_CHANNEL_NAMES: list = [ch[0] for ch in DRIVER_INPUT_CHANNELS]
