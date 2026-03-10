"""
LapTimeSimulator — Streamlit Interface

Modes:
    Copa Truck  : gear_min=4, HDF5 track or Interlagos GPS-ref
    Porsche GT3 : gear_min=1, full fleet (991.1 / 991.2 / 992.1) + VehicleSetup

Run from project root:
    streamlit run src/visualization/interface.py
"""
# isort: skip_file
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import logging
import time
import os
import sys
import hashlib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from src.simulation.lap_time_solver import run_bicycle_model  # noqa: E402
from src.tracks.generate_br_tracks import build_interlagos_real  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Optional imports — Copa Truck legacy objects
try:
    from src.tracks.hdf5 import CircuitHDF5Reader
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    from src.vehicle.parameters import copa_truck_2dof_default
    TRUCK_AVAILABLE = True
except ImportError:
    TRUCK_AVAILABLE = False

# Porsche fleet
try:
    from src.vehicle.fleet import get_vehicle_by_id, list_vehicle_ids
    from src.vehicle.setup import VehicleSetup, apply_setup
    FLEET_AVAILABLE = True
except ImportError:
    FLEET_AVAILABLE = False

DATA_PATH = str(BASE_DIR / "tracks")
RESULTS_PATH = str(BASE_DIR / "src" / "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_laptime(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:06.3f}"


def _solver_cache_key(params_dict: dict, circuit, config: dict) -> str:
    """Build a deterministic hash from solver inputs for caching."""
    parts: list[str] = []
    for k in sorted(params_dict.keys()):
        v = params_dict[k]
        if isinstance(v, np.ndarray):
            parts.append(f"{k}={v.tobytes().hex()[:32]}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{k}={str(v)}")
        else:
            parts.append(f"{k}={v}")
    parts.append(f"cfg={json.dumps(config, sort_keys=True)}")
    parts.append(f"cx={circuit.centerline_x.tobytes().hex()[:32]}")
    parts.append(f"cy={circuit.centerline_y.tobytes().hex()[:32]}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


_solver_result_cache: dict[str, dict] = {}


def _cached_solver(params_dict, circuit, config, save_csv=False, out_path=None):
    """Run solver with in-memory cache (skip re-computation for same inputs)."""
    key = _solver_cache_key(params_dict, circuit, config)
    if key in _solver_result_cache:
        result = _solver_result_cache[key]
        # Still save CSV if requested but result was cached
        if save_csv and out_path:
            _save_result_csv(result, out_path)
        return result
    result = run_bicycle_model(
        params_dict=params_dict, circuit=circuit, config=config,
        save_csv=save_csv, out_path=out_path,
    )
    _solver_result_cache[key] = result
    return result


def _save_result_csv(result: dict, path: str):
    """Write a cached result to CSV (same columns as solver)."""
    df = pd.DataFrame({
        "Distance": result["distance"],
        "Time": result["time"],
        "Speed": result["v_profile"] * 3.6,
        "G_Long": result["a_long"] / 9.81,
        "G_Lat": result["a_lat"] / 9.81,
        "Gear": result["gear"],
        "Engine_RPM": result["rpm"],
    })
    df.to_csv(path, index=False)


@st.cache_data
def _load_hdf5(path: str):
    circuit, meta = CircuitHDF5Reader(path).read_circuit()
    x_c = -(circuit.centerline_y - circuit.centerline_y[0])
    y_c = circuit.centerline_x - circuit.centerline_x[0]
    left_x = -(circuit.left_boundary_y - circuit.centerline_y[0])
    left_y = circuit.left_boundary_x - circuit.centerline_x[0]
    right_x = -(circuit.right_boundary_y - circuit.centerline_y[0])
    right_y = circuit.right_boundary_x - circuit.centerline_x[0]
    return circuit, meta, dict(x_c=x_c, y_c=y_c,
                               left_x=left_x, left_y=left_y,
                               right_x=right_x, right_y=right_y)


@st.cache_data
def _load_interlagos_real():
    circuit = build_interlagos_real(n_points=4000)
    meta = {
        "name":   "Interlagos — GPS ref.",
        "length": float(np.sum(np.sqrt(
            np.diff(circuit.centerline_x)**2 +
            np.diff(circuit.centerline_y)**2
        ))),
    }
    plot_data = dict(
        x_c=circuit.centerline_x, y_c=circuit.centerline_y,
        left_x=circuit.left_boundary_x, left_y=circuit.left_boundary_y,
        right_x=circuit.right_boundary_x, right_y=circuit.right_boundary_y,
    )
    return circuit, meta, plot_data


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "vehicle_mode":   "Copa Truck",
        "vehicle_id":     "porsche_991_1",
        "vehicle_params": None,
        "solver_dict":    None,
        "setup":          None,
        "circuit":        None,
        "circuit_meta":   None,
        "resultados_prontos": False,
        "resultados":     None,
        "csv_path":       None,
        "all_results":    [],   # list of dicts for setup comparison
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# PAGE: Parameters
# ---------------------------------------------------------------------------
def parametros_veiculo_page():
    st.header("🚗 Vehicle Parameters")

    mode = st.radio(
        "Simulation mode:",
        ["Copa Truck", "Porsche GT3 Cup"],
        horizontal=True,
        key="vehicle_mode",
    )

    # ---- Copa Truck --------------------------------------------------------
    if mode == "Copa Truck":
        if not TRUCK_AVAILABLE:
            st.error(
                "`copa_truck_2dof_default` not found. Check src/vehicle/parameters.py.")
            return
        if st.session_state.vehicle_params is None or not hasattr(
            st.session_state.vehicle_params, 'manufacturer'
        ):
            st.session_state.vehicle_params = copa_truck_2dof_default()

        vp = st.session_state.vehicle_params
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Manufacturer", vp.manufacturer)
        with col2:
            st.metric("Year", vp.year)
        with col3:
            st.metric("Power", f"{vp.engine.max_power/1000:.0f} kW")

        st.markdown("---")
        if st.radio("Customize?", ["No", "Yes"], horizontal=True) == "Yes":
            sec = st.radio("Section:", ["Mass/Geometry", "Tire", "Engine",
                                        "Transmission", "Brake", "Aerodynamics"],
                           horizontal=True)
            if sec == "Mass/Geometry":
                vp.mass_geometry.mass = st.number_input("Mass (kg)", 3500.0, 9000.0,
                                                        float(vp.mass_geometry.mass))
                wb = st.number_input("Wheelbase (m)", 3.8, 5.5,
                                     float(vp.mass_geometry.wheelbase))
                vp.mass_geometry.lf = st.number_input("CG → front (m)", 1.0, 3.0,
                                                      float(vp.mass_geometry.lf))
                vp.mass_geometry.lr = wb - vp.mass_geometry.lf
                vp.mass_geometry.wheelbase = wb
            elif sec == "Tire":
                vp.tire.friction_coefficient = st.number_input(
                    "μ (Base Cold Grip)", 0.8, 1.5,
                    float(vp.tire.friction_coefficient))
            elif sec == "Engine":
                vp.engine.max_power = st.number_input(
                    "Power (kW)", 300.0, 900.0,
                    float(vp.engine.max_power) / 1000.0) * 1000.0
                vp.engine.max_torque = st.number_input(
                    "Torque (Nm)", 2000.0, 6500.0,
                    float(vp.engine.max_torque))
            elif sec == "Transmission":
                vp.transmission.num_gears = st.slider(
                    "Gears", 6, 16, int(vp.transmission.num_gears))
                vp.transmission.final_drive_ratio = st.number_input(
                    "Final drive", 2.8, 7.5,
                    float(vp.transmission.final_drive_ratio))
            elif sec == "Brake":
                vp.brake.max_deceleration = st.slider(
                    "Max decel (m/s²)", 3.0, 10.0,
                    float(vp.brake.max_deceleration))
            elif sec == "Aerodynamics":
                vp.aero.drag_coefficient = st.number_input(
                    "Cd", 0.45, 1.20, float(vp.aero.drag_coefficient))
                vp.aero.frontal_area = st.number_input(
                    "Frontal area (m²)", 7.0, 10.0,
                    float(vp.aero.frontal_area))
        st.session_state.vehicle_params = vp

    # ---- Porsche GT3 Cup --------------------------------------------------
    else:
        if not FLEET_AVAILABLE:
            st.error("`src/vehicle/fleet` not found. Run git pull.")
            return

        vehicle_ids = list_vehicle_ids() if hasattr(
            sys.modules.get('src.vehicle.fleet', object()), 'list_vehicle_ids'
        ) else ["porsche_991_1", "porsche_991_2", "porsche_992_1"]

        vid = st.selectbox("Vehicle:", vehicle_ids,
                           index=vehicle_ids.index(st.session_state.vehicle_id)
                           if st.session_state.vehicle_id in vehicle_ids else 0)
        st.session_state.vehicle_id = vid
        base = get_vehicle_by_id(vid)

        st.markdown("---")
        st.subheader("Setup configuration")
        col1, col2, col3, col4, col5 = st.columns(5)
        arb_f = col1.slider("ARB Front",  1, 7, 4)
        arb_r = col2.slider("ARB Rear",   1, 7, 4)
        wing = col3.slider("Wing pos.",  1, 9, 5)
        pressure = col4.number_input("Tyre P (bar)", 1.4, 2.4, 1.8, step=0.05)
        bias = col5.slider("Brake bias", -2.0, 0.0, -1.0, step=0.5)

        setup = VehicleSetup(
            arb_front=arb_f, arb_rear=arb_r, wing_position=wing,
            tyre_pressure=float(pressure), brake_bias=float(bias),
            setup_name=f"ARB{arb_f}/{arb_r}_W{wing}",
        )
        params = apply_setup(base, setup)
        st.session_state.vehicle_params = params
        st.session_state.setup = setup

        d = setup.to_dict()
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        col_i1.metric("ARB front stiffness",
                      f"{d['arb_front_stiffness_nm_rad']/1000:.0f} kNm/rad")
        col_i2.metric("ARB rear stiffness",
                      f"{d['arb_rear_stiffness_nm_rad']/1000:.0f} kNm/rad")
        col_i3.metric("ΔCd", f"{d['wing_delta_cd']:+.3f}")
        col_i4.metric("Balance", d['handling_balance'])


# ---------------------------------------------------------------------------
# PAGE: Track
# ---------------------------------------------------------------------------
def pista_page():
    st.header("🗭️ Track")

    track_source = st.radio(
        "Track source:",
        ["Interlagos (GPS real)", "HDF5 file"],
        horizontal=True,
    )

    if track_source == "Interlagos (GPS real)":
        circuit, meta, plot_data = _load_interlagos_real()
    else:
        if not HDF5_AVAILABLE:
            st.error("HDF5 reader not available.")
            return
        if not os.path.isdir(DATA_PATH):
            st.warning(f"Pasta de pistas não encontrada: {DATA_PATH}")
            return
        pistas = [f for f in os.listdir(DATA_PATH) if f.endswith('.hdf5')]
        if not pistas:
            st.warning(f'Nenhuma pista .hdf5 encontrada em {DATA_PATH}!')
            return
        sel = st.selectbox("Select:", pistas)
        circuit, meta, plot_data = _load_hdf5(os.path.join(DATA_PATH, sel))

    st.session_state.circuit = circuit
    st.session_state.circuit_meta = meta

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data['x_c'],    y=plot_data['y_c'],
                             mode="lines", name="Centerline",
                             line=dict(color="royalblue", width=2)))
    fig.add_trace(go.Scatter(x=plot_data['left_x'], y=plot_data['left_y'],
                             mode="lines", name="Left",
                             line=dict(color='limegreen', dash='dot', width=1)))
    fig.add_trace(go.Scatter(x=plot_data['right_x'], y=plot_data['right_y'],
                             mode="lines", name="Right",
                             line=dict(color='tomato', dash='dot', width=1)))
    fig.update_layout(title=meta['name'], xaxis_title="x (m)", yaxis_title="y (m)",
                      margin=dict(l=0, r=0, t=35, b=0), height=450)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, width='stretch')
    st.success(f"✓ {meta['name']} | {meta['length']:.0f} m")


# ---------------------------------------------------------------------------
# PAGE: Simulation
# ---------------------------------------------------------------------------
def simulacao_page():
    st.header("▶️ Simulation")

    if st.session_state.circuit is None:
        st.warning("⚠️ Select a track in the 'Track' tab first.")
        return
    if st.session_state.vehicle_params is None:
        st.warning("⚠️ Configure a vehicle in the 'Parameters' tab first.")
        return

    mode = st.session_state.get("vehicle_mode", "Copa Truck")
    st.info(
        f"✓ Track: **{st.session_state.circuit_meta['name']}** | Mode: **{mode}**")

    col_play, col_reset = st.columns(2)
    with col_reset:
        if st.button("🗑️ Clear results", width='stretch'):
            st.session_state.resultados_prontos = False
            st.session_state.resultados = None
            st.session_state.all_results = []
            st.rerun()

    with col_play:
        if st.button("▶ Simulate", width='stretch', type="primary"):
            with st.spinner("🔄 Running GGV solver..."):
                vp = st.session_state.vehicle_params

                # Build solver_dict
                if hasattr(vp, 'to_solver_dict'):
                    params_dict = vp.to_solver_dict()
                else:
                    # Copa Truck legacy VehicleParams
                    params_dict = vp.to_solver_dict()

                params_dict.setdefault("track_width", 2.45)

                # Solver config
                gear_min = 1 if mode == "Porsche GT3 Cup" else 4
                solver_config = {"gear_min": gear_min}
                # coef_aderencia intentionally NOT set here — mu comes from TireParams

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pista_nome = st.session_state.circuit_meta["name"].replace(" ", "_")[
                    :20]
                csv_path = os.path.join(
                    RESULTS_PATH, f"lap_{pista_nome}_{timestamp}.csv"
                )

                t0 = time.perf_counter()
                try:
                    result = _cached_solver(
                        params_dict=params_dict,
                        circuit=st.session_state.circuit,
                        config=solver_config,
                        save_csv=True,
                        out_path=csv_path,
                    )
                    elapsed = time.perf_counter() - t0

                    st.session_state.resultados = result
                    st.session_state.csv_path = csv_path
                    st.session_state.resultados_prontos = True

                    # Store for multi-setup comparison
                    label = (
                        setup.setup_name
                        if setup else
                        getattr(getattr(vp, 'name', None),
                                '__str__', lambda: mode)()
                    )
                    st.session_state.all_results.append({
                        "label":      label,
                        "lap_time":   result["lap_time"],
                        "vmax":       float(np.max(result["v_profile"])) * 3.6,
                        "vmean":      float(np.mean(result["v_profile"])) * 3.6,
                        "fuel_L":     float(result["consumo"][-1]),
                        "tyre_temp":  float(result["temp_pneu"][-1]),
                        "result_obj": result,
                    })

                    st.success(
                        f"✓ Lap: **{fmt_laptime(result['lap_time'])}** — "
                        f"Vmax: **{float(np.max(result['v_profile'])*3.6):.1f} km/h** — "
                        f"compute: {elapsed:.3f}s"
                    )
                except Exception as exc:
                    import traceback
                    st.error(f"Solver error: {exc}")
                    st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# PAGE: Results
# ---------------------------------------------------------------------------
def resultados_page():
    st.header("🏁 Results & Telemetry")

    if not st.session_state.get("resultados_prontos", False):
        st.warning("⚠️ Run a simulation first.")
        return

    res = st.session_state.resultados
    csv_file = st.session_state.csv_path
    circuit = st.session_state.circuit

    g = 9.81
    v_kmh = res['v_profile'] * 3.6
    alon_g = res['a_long'] / g
    alat_g = res['a_lat'] / g
    dist = res['distance']

    tempo_total = res['time'][-1]
    dt_arr = np.diff(np.append([0], res['time']))
    time_wot = float(np.sum((alon_g > 0.05) * dt_arr))
    time_brake = float(np.sum((alon_g < -0.1) * dt_arr))
    max_lat_g = float(np.max(np.abs(alat_g)))
    max_roll = float(np.max(np.abs(res.get('roll_angle_profile', [0]))))
    t_pneu_fim = float(res['temp_pneu'][-1])
    p_pneu_fim = float(res['pressao_pneu'][-1])

    # --- KPIs ---
    st.subheader("🏁 Performance KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lap Time",    fmt_laptime(res['lap_time']))
    c2.metric("Avg Speed",   f"{float(np.mean(v_kmh)):.1f} km/h")
    c3.metric("Max Speed",   f"{float(np.max(v_kmh)):.1f} km/h")
    c4.metric("WOT %",       f"{(time_wot/tempo_total)*100:.1f} %")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Peak Lat G",    f"{max_lat_g:.2f} G")
    c6.metric("Peak Brake G",  f"{float(np.min(alon_g)):.2f} G")
    c7.metric("Peak Accel G",  f"{float(np.max(alon_g)):.2f} G")
    c8.metric("Braking %",     f"{(time_brake/tempo_total)*100:.1f} %")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Peak Roll",     f"{max_roll:.2f} °")
    c10.metric("Tyre Temp end", f"{t_pneu_fim:.1f} °C")
    c11.metric("Tyre Press end", f"{p_pneu_fim:.2f} bar")
    c12.metric("Fuel Used",     f"{float(np.max(res['consumo'])):.3f} L")

    st.markdown("---")

    # --- Download CSV ---
    if csv_file and os.path.exists(csv_file):
        with open(csv_file, "rb") as f:
            st.download_button(
                label="📥 Download full telemetry CSV (MoTeC / PiToolbox)",
                data=f,
                file_name=f"LapSim_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
            )

    st.markdown("---")
    st.subheader("🗺️ Speed Map")

    # Colour-coded speed map on circuit centreline
    x_c = circuit.centerline_x
    y_c = circuit.centerline_y
    n = min(len(x_c), len(v_kmh))
    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=x_c[:n], y=y_c[:n], mode='markers',
        marker=dict(
            size=3,
            color=v_kmh[:n],
            colorscale='RdYlGn',
            colorbar=dict(title='Speed (km/h)'),
            cmin=float(np.min(v_kmh[:n])),
            cmax=float(np.max(v_kmh[:n])),
        ),
        name='Speed',
    ))
    fig_map.update_layout(
        title='Speed Map — colour = velocity',
        xaxis_title='x (m)', yaxis_title='y (m)',
        height=500, margin=dict(l=0, r=0, t=35, b=0),
    )
    fig_map.update_yaxes(scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig_map, width='stretch')

    st.markdown("---")
    st.subheader("📈 Dynamics Channels")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=dist, y=v_kmh, mode='lines',
                                   name='Speed', line=dict(color='royalblue', width=2)))
        fig_v.update_layout(title='Speed (km/h)', height=280,
                            margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_v, width='stretch')

    with col_g2:
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=dist, y=alat_g, mode='lines',
                                   name='Lat G', line=dict(color='tomato', width=2)))
        fig_a.add_trace(go.Scatter(x=dist, y=alon_g, mode='lines',
                                   name='Long G', line=dict(color='seagreen', width=2)))
        fig_a.update_layout(title='Longitudinal & Lateral G', height=280,
                            margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_a, width='stretch')

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=dist, y=res['temp_pneu'], mode='lines',
                                      name='Tyre Temp',
                                      line=dict(color='darkorange', width=2)))
        fig_temp.add_hline(y=95.0, line_dash='dash', line_color='green',
                           annotation_text='Optimum')
        fig_temp.update_layout(title='Tyre Temperature (°C)', height=280,
                               margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_temp, width='stretch')

    with col_g4:
        fig_press = go.Figure()
        fig_press.add_trace(go.Scatter(x=dist, y=res['pressao_pneu'], mode='lines',
                                       name='Tyre Press',
                                       line=dict(color='teal', width=2)))
        fig_press.update_layout(title='Tyre Pressure (bar)', height=280,
                                margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_press, width='stretch')

    col_g5, col_g6 = st.columns(2)
    with col_g5:
        fig_rpm = go.Figure()
        fig_rpm.add_trace(go.Scatter(x=dist, y=res['rpm'], mode='lines',
                                     name='RPM', line=dict(color='purple', width=2)))
        fig_rpm.add_trace(go.Scatter(x=dist, y=res['gear'] * 1000, mode='lines',
                                     name='Gear ×1000', line=dict(color='gray',
                                                                  width=1, dash='dot')))
        fig_rpm.update_layout(title='Engine RPM + Gear (×1000)', height=280,
                              margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_rpm, width='stretch')

    with col_g6:
        fig_ggv = go.Figure()
        fig_ggv.add_trace(go.Scatter(
            x=alat_g, y=alon_g, mode='markers',
            marker=dict(size=3, color=v_kmh, colorscale='Viridis',
                        colorbar=dict(title='km/h')),
        ))
        fig_ggv.update_layout(
            title='GGV Diagram', xaxis_title='Lat G', yaxis_title='Long G',
            height=400, yaxis_range=[-1.5, 1.5], xaxis_range=[-1.5, 1.5],
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig_ggv.update_yaxes(scaleanchor='x', scaleratio=1)
        st.plotly_chart(fig_ggv, width='stretch')

    # --- Roll & Slip ---
    col_g7, col_g8 = st.columns(2)
    with col_g7:
        if 'roll_angle_profile' in res:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=dist, y=res['roll_angle_profile'], mode='lines',
                name='Roll angle', line=dict(color='sienna', width=2)))
            fig_roll.update_layout(title='Cabin Roll Angle (°)', height=280,
                                   margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_roll, width='stretch')
    with col_g8:
        slip_data = res.get('front_slip_angle_deg', np.zeros(len(dist)))
        fig_slip = go.Figure()
        fig_slip.add_trace(go.Scatter(
            x=dist, y=slip_data, mode='lines',
            name='Slip angle', line=dict(color='darkviolet', width=2)))
        fig_slip.update_layout(title='Front Slip Angle (°)', height=280,
                               margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_slip, width='stretch')

    # --- Driver Inputs ---
    st.markdown("---")
    st.subheader("🎮 Driver Inputs")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_pedals = go.Figure()
        fig_pedals.add_trace(go.Scatter(
            x=dist, y=res.get('throttle_pct', np.zeros(len(dist))),
            mode='lines', name='Throttle %',
            line=dict(color='limegreen', width=2)))
        fig_pedals.add_trace(go.Scatter(
            x=dist, y=res.get('brake_pct', np.zeros(len(dist))),
            mode='lines', name='Brake %',
            line=dict(color='red', width=2)))
        fig_pedals.update_layout(
            title='Throttle & Brake (%)', height=280,
            yaxis_title='%', xaxis_title='Distance (m)',
            margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pedals, width='stretch')

    with col_d2:
        fig_steer = go.Figure()
        fig_steer.add_trace(go.Scatter(
            x=dist, y=res.get('steering_deg', np.zeros(len(dist))),
            mode='lines', name='Steering',
            line=dict(color='dodgerblue', width=2)))
        fig_steer.update_layout(
            title='Steering Angle (°)', height=280,
            yaxis_title='deg', xaxis_title='Distance (m)',
            margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_steer, width='stretch')

    # --- Porsche Data Analysis Sections (ENG170914 / ENG210319) ---

    # ── Sector Timing ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏁 Sector Timing")
    track_len = float(dist[-1])
    n_sectors = st.slider("Number of sectors", 3, 12, 3, key="n_sectors")
    sector_boundaries = np.linspace(0, track_len, n_sectors + 1)
    sector_rows = []
    for s_idx in range(n_sectors):
        s_start, s_end = sector_boundaries[s_idx], sector_boundaries[s_idx + 1]
        mask = (dist >= s_start) & (dist < s_end)
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        t_sector = res['time'][idxs[-1]] - res['time'][idxs[0]]
        v_avg_s = float(np.mean(v_kmh[mask]))
        v_min_s = float(np.min(v_kmh[mask]))
        v_max_s = float(np.max(v_kmh[mask]))
        sector_rows.append({
            "Sector": f"S{s_idx+1}",
            "From (m)": f"{s_start:.0f}",
            "To (m)": f"{s_end:.0f}",
            "Time": fmt_laptime(t_sector),
            "Time (s)": f"{t_sector:.3f}",
            "V avg (km/h)": f"{v_avg_s:.1f}",
            "V min (km/h)": f"{v_min_s:.1f}",
            "V max (km/h)": f"{v_max_s:.1f}",
        })
    if sector_rows:
        st.dataframe(pd.DataFrame(sector_rows), width='stretch')

    # ── Braking Zone Analysis ────────────────────────────────────
    st.markdown("---")
    st.subheader("🛑 Braking Zone Analysis")
    brake_threshold = -0.15  # G threshold to detect braking
    in_brake = False
    brake_zones = []
    bz_start = 0
    for i in range(len(alon_g)):
        if alon_g[i] < brake_threshold and not in_brake:
            in_brake = True
            bz_start = i
        elif (alon_g[i] >= brake_threshold or i == len(alon_g) - 1) and in_brake:
            in_brake = False
            bz_end = i
            if bz_end - bz_start > 5:  # filter out noise
                d_brake = dist[bz_end] - dist[bz_start]
                v_entry = float(v_kmh[bz_start])
                v_exit = float(v_kmh[bz_end])
                peak_decel = float(np.min(alon_g[bz_start:bz_end+1]))
                t_brake = res['time'][bz_end] - res['time'][bz_start]
                brake_zones.append({
                    "Zone": len(brake_zones) + 1,
                    "Start (m)": f"{dist[bz_start]:.0f}",
                    "Distance (m)": f"{d_brake:.0f}",
                    "V entry (km/h)": f"{v_entry:.0f}",
                    "V exit (km/h)": f"{v_exit:.0f}",
                    "ΔV (km/h)": f"{v_entry - v_exit:.0f}",
                    "Peak G": f"{peak_decel:.2f}",
                    "Duration (s)": f"{t_brake:.2f}",
                })
    if brake_zones:
        st.dataframe(pd.DataFrame(brake_zones), width='stretch')
        st.caption(f"Total braking zones detected: **{len(brake_zones)}** "
                   f"(threshold: {brake_threshold} G)")
    else:
        st.info("No significant braking zones detected.")

    # ── Friction Utilization ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🔵 Friction Utilization")
    mu_used = np.sqrt(alon_g**2 + alat_g**2)
    mu_available = np.full_like(mu_used,
                                float(res.get('grip_mult', np.ones(1))[-1])
                                * (st.session_state.vehicle_params.tire.friction_coefficient
                                   if hasattr(st.session_state.vehicle_params, 'tire')
                                   else 1.1))
    mu_util_pct = np.clip(mu_used / mu_available * 100, 0, 100)

    col_mu1, col_mu2 = st.columns(2)
    with col_mu1:
        fig_mu = go.Figure()
        fig_mu.add_trace(go.Scatter(
            x=dist, y=mu_util_pct, mode='lines',
            name='μ utilization', line=dict(color='coral', width=2)))
        fig_mu.add_hline(y=90, line_dash='dash', line_color='green',
                         annotation_text='90% target')
        fig_mu.update_layout(title='Friction Utilization (%)', height=280,
                             yaxis_title='%', xaxis_title='Distance (m)',
                             margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mu, width='stretch')

    with col_mu2:
        # Friction circle filled
        fig_fc = go.Figure()
        theta = np.linspace(0, 2 * np.pi, 100)
        mu_peak = float(np.mean(mu_available))
        fig_fc.add_trace(go.Scatter(
            x=mu_peak * np.cos(theta), y=mu_peak * np.sin(theta),
            mode='lines', name='Available μ',
            line=dict(color='gray', dash='dash', width=1)))
        fig_fc.add_trace(go.Scatter(
            x=alat_g, y=alon_g, mode='markers',
            marker=dict(size=2, color=v_kmh, colorscale='Viridis',
                        colorbar=dict(title='km/h')),
            name='Actual'))
        fig_fc.update_layout(
            title='Friction Circle (G-G)',
            xaxis_title='Lat G', yaxis_title='Long G',
            height=350, margin=dict(l=0, r=0, t=30, b=0))
        fig_fc.update_yaxes(scaleanchor='x', scaleratio=1)
        st.plotly_chart(fig_fc, width='stretch')

    mu_avg = float(np.mean(mu_util_pct))
    mu_p90 = float(np.percentile(mu_util_pct, 90))
    c_mu1, c_mu2, c_mu3 = st.columns(3)
    c_mu1.metric("Avg μ Utilization", f"{mu_avg:.1f} %")
    c_mu2.metric("P90 μ Utilization", f"{mu_p90:.1f} %")
    c_mu3.metric("Peak Combined G", f"{float(np.max(mu_used)):.2f} G")

    # ── RPM / Power Histogram ────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 RPM & Gear Distribution")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        rpm_data = res['rpm']
        rpm_data = rpm_data[rpm_data > 0]
        fig_rpm_hist = go.Figure()
        fig_rpm_hist.add_trace(go.Histogram(
            x=rpm_data, nbinsx=30,
            marker_color='mediumpurple', name='RPM'))
        fig_rpm_hist.update_layout(
            title='RPM Distribution', height=300,
            xaxis_title='RPM', yaxis_title='Count',
            margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_rpm_hist, width='stretch')

    with col_h2:
        gear_data = res['gear']
        gear_counts = pd.Series(gear_data).value_counts().sort_index()
        fig_gear = go.Figure(go.Bar(
            x=[f"Gear {int(g)}" for g in gear_counts.index],
            y=gear_counts.values / len(gear_data) * 100,
            marker_color='steelblue',
            text=[f"{v:.1f}%" for v in gear_counts.values /
                  len(gear_data) * 100],
            textposition='outside',
        ))
        fig_gear.update_layout(
            title='Gear Usage (%)', height=300,
            yaxis_title='% of lap', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_gear, width='stretch')

    # ── Gap Analysis (multi-run) ─────────────────────────────────
    all_res = st.session_state.get('all_results', [])
    if len(all_res) > 1:
        st.markdown("---")
        st.subheader("📈 Gap Analysis — Distance Domain")
        st.caption("Time difference (Δt) between the first run and subsequent "
                   "runs at each distance point.")

        ref = all_res[0]
        ref_dist = ref['result_obj']['distance']
        ref_time = ref['result_obj']['time']
        colors_gap = px.colors.qualitative.Plotly

        fig_gap = go.Figure()
        for idx, run in enumerate(all_res[1:], 1):
            run_dist = run['result_obj']['distance']
            run_time = run['result_obj']['time']
            # Interpolate to common distance grid
            common_dist = np.linspace(0, min(ref_dist[-1], run_dist[-1]), 500)
            ref_t_interp = np.interp(common_dist, ref_dist, ref_time)
            run_t_interp = np.interp(common_dist, run_dist, run_time)
            delta_t = run_t_interp - ref_t_interp
            fig_gap.add_trace(go.Scatter(
                x=common_dist, y=delta_t, mode='lines',
                name=f'{run["label"]} vs {ref["label"]}',
                line=dict(color=colors_gap[idx % len(colors_gap)], width=2),
            ))
        fig_gap.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_gap.update_layout(
            title=f'Δt vs "{ref["label"]}" (+ = slower)',
            xaxis_title='Distance (m)', yaxis_title='Δ time (s)',
            height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_gap, width='stretch')

    # --- HTML Export ---
    st.markdown("---")
    st.subheader("📄 HTML Report Export")
    if st.button("📄 Export all charts as HTML", width='stretch'):
        figs_html = [
            fig_map, fig_v, fig_a, fig_temp, fig_press, fig_rpm, fig_ggv,
        ]
        fig_names = [
            "Speed Map", "Speed", "Long & Lat G", "Tyre Temp",
            "Tyre Pressure", "RPM & Gear", "GGV Diagram",
        ]
        html_parts = [
            "<html><head><meta charset='utf-8'>"
            "<title>LapTimeSimulator Report</title></head><body>"
            f"<h1>LapTimeSimulator Report — {fmt_laptime(res['lap_time'])}</h1>"
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        for name, fig in zip(fig_names, figs_html):
            html_parts.append(f"<h2>{name}</h2>")
            html_parts.append(pio.to_html(
                fig, full_html=False, include_plotlyjs='cdn'))
        html_parts.append("</body></html>")
        full_html = "\n".join(html_parts)
        st.download_button(
            label="📥 Download HTML report",
            data=full_html, file_name="lap_report.html",
            mime="text/html",
        )

    # --- Setup comparison table ---
    all_res = st.session_state.get('all_results', [])
    if len(all_res) > 1:
        st.markdown("---")
        st.subheader("📊 Setup Comparison")
        rows = []
        for r in all_res:
            rows.append({
                "Setup":        r['label'],
                "Lap Time":     fmt_laptime(r['lap_time']),
                "Lap Time (s)": f"{r['lap_time']:.3f}",
                "Vmax (km/h)":  f"{r['vmax']:.1f}",
                "Vmean (km/h)": f"{r['vmean']:.1f}",
                "Fuel (L)":     f"{r['fuel_L']:.3f}",
                "T_tyre (\u00b0C)":  f"{r['tyre_temp']:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch')

        # Delta bar chart
        base_t = all_res[0]['lap_time']
        labels = [r['label'] for r in all_res]
        deltas = [r['lap_time'] - base_t for r in all_res]
        fig_delta = go.Figure(go.Bar(
            x=labels, y=deltas,
            marker_color=['green' if d <= 0 else 'red' for d in deltas],
            text=[f"{d:+.3f}s" for d in deltas], textposition='outside',
        ))
        fig_delta.update_layout(
            title=f'Δ Lap Time vs. "{labels[0]}" (s)',
            yaxis_title='Δ lap time (s)', height=350,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_delta, width='stretch')


# ---------------------------------------------------------------------------
# PAGE: Compare (CSV Upload)
# ---------------------------------------------------------------------------
_KNOWN_COLUMNS = {
    # Current solver CSV columns → internal key mapping
    "Distance": "distance", "Time": "time", "Speed": "speed_kmh",
    "Engine_RPM": "rpm", "Gear": "gear",
    "G_Long": "g_long", "G_Lat": "g_lat",
    "Throttle_Pos": "throttle_pct", "Brake_Press": "brake_pct",
    "Steering_Angle_deg": "steering_deg", "Roll_Angle_deg": "roll_deg",
    "Front_Slip_Angle_deg": "slip_deg",
    "Fuel_Cons_Accum_L": "fuel_l", "Tyre_Temp_C": "tyre_temp",
    "Tyre_Press_bar": "tyre_press", "Corner_Radius_m": "radius",
    # Legacy CSV columns
    "distance_m": "distance", "v_kmh": "speed_kmh",
    "a_long_ms2": "g_long", "a_lat_ms2": "g_lat",
    "time_s": "time", "temp_pneu_c": "tyre_temp", "consumo_l": "fuel_l",
    "radius_m": "radius",
}


def _parse_uploaded_csv(uploaded) -> dict | None:
    """Parse an uploaded CSV and return a normalised dict of arrays."""
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    out: dict[str, np.ndarray] = {}
    for col in df.columns:
        key = _KNOWN_COLUMNS.get(col, col.lower().replace(" ", "_"))
        out[key] = df[col].values
    return out


def compare_page():
    st.header("📊 Compare — CSV Upload")
    st.caption(
        "Upload one or more CSV telemetry files (exported from the simulator "
        "or MoTeC / PiToolbox) and overlay them against the current simulation."
    )

    uploaded_files = st.file_uploader(
        "Drop CSV files here", type=["csv"], accept_multiple_files=True,
    )

    datasets: list[tuple[str, dict]] = []

    # Current sim data (if available)
    res = st.session_state.get("resultados")
    if res is not None:
        datasets.append(("Sim (current)", {
            "distance": res["distance"],
            "speed_kmh": res["v_profile"] * 3.6,
            "g_long": res["a_long"] / 9.81,
            "g_lat": res["a_lat"] / 9.81,
            "throttle_pct": res.get("throttle_pct", np.zeros(len(res["distance"]))),
            "brake_pct": res.get("brake_pct", np.zeros(len(res["distance"]))),
            "steering_deg": res.get("steering_deg", np.zeros(len(res["distance"]))),
            "tyre_temp": res.get("temp_pneu", np.zeros(len(res["distance"]))),
        }))

    if uploaded_files:
        for uf in uploaded_files:
            parsed = _parse_uploaded_csv(uf)
            if parsed is None:
                st.warning(f"⚠️ Could not parse **{uf.name}**")
                continue
            datasets.append((uf.name, parsed))

    if len(datasets) < 1:
        st.info("Upload at least one CSV file, or run a simulation first.")
        return

    colors = px.colors.qualitative.Plotly
    channel_defs = [
        ("speed_kmh", "Speed (km/h)"),
        ("g_long",    "Longitudinal G"),
        ("g_lat",     "Lateral G"),
        ("throttle_pct", "Throttle (%)"),
        ("brake_pct", "Brake (%)"),
        ("steering_deg", "Steering Angle (°)"),
        ("tyre_temp", "Tyre Temperature (°C)"),
    ]

    for key, title in channel_defs:
        has_data = any(key in d for _, d in datasets)
        if not has_data:
            continue
        fig = go.Figure()
        for idx, (name, data) in enumerate(datasets):
            if key not in data:
                continue
            x = data.get("distance", np.arange(len(data[key])))
            fig.add_trace(go.Scatter(
                x=x, y=data[key], mode="lines",
                name=name, line=dict(color=colors[idx % len(colors)], width=2),
            ))
        fig.update_layout(
            title=title, xaxis_title="Distance (m)", height=300,
            margin=dict(l=0, r=0, t=35, b=0),
        )
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# PAGE: Setup Optimization
# ---------------------------------------------------------------------------
def optimization_page():
    st.header("🔧 Setup Optimization")

    if not FLEET_AVAILABLE:
        st.error("Fleet module not available.")
        return
    if st.session_state.circuit is None:
        st.warning("⚠️ Select a track in the 'Track' tab first.")
        return

    mode = st.session_state.get("vehicle_mode", "Copa Truck")
    if mode != "Porsche GT3 Cup":
        st.warning(
            "Setup optimisation is only available in **Porsche GT3 Cup** mode.")
        return

    st.caption(
        "Grid-search over ARB front, ARB rear and Wing position to find the "
        "fastest setup combination. Tyre pressure and brake bias are held constant."
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pressure = st.number_input("Tyre Pressure (bar)", 1.4, 2.4, 1.8, step=0.05,
                                   key="opt_pressure")
    with col_p2:
        bias = st.slider("Brake Bias", -2.0, 0.0, -1.0, step=0.5,
                         key="opt_bias")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        arb_f_range = st.slider("ARB Front range", 1, 7,
                                (1, 7), key="opt_arb_f")
    with col_r2:
        arb_r_range = st.slider("ARB Rear range", 1, 7,
                                (1, 7), key="opt_arb_r")
    with col_r3:
        wing_range = st.slider("Wing range", 1, 9, (1, 9), key="opt_wing")

    arb_f_vals = list(range(arb_f_range[0], arb_f_range[1] + 1))
    arb_r_vals = list(range(arb_r_range[0], arb_r_range[1] + 1))
    wing_vals = list(range(wing_range[0], wing_range[1] + 1))
    total = len(arb_f_vals) * len(arb_r_vals) * len(wing_vals)

    st.info(f"Total combinations: **{total}** (ARB-F: {len(arb_f_vals)} × "
            f"ARB-R: {len(arb_r_vals)} × Wing: {len(wing_vals)})")

    if st.button("🚀 Run Optimisation", width='stretch', type="primary"):
        base = get_vehicle_by_id(st.session_state.vehicle_id)
        circuit = st.session_state.circuit
        results_opt: list[dict] = []

        bar = st.progress(0, text="Running grid search...")
        t0 = time.perf_counter()

        for idx, (af, ar, w) in enumerate(product(arb_f_vals, arb_r_vals, wing_vals)):
            bar.progress((idx + 1) / total,
                         text=f"Combo {idx+1}/{total} — ARB {af}/{ar} Wing {w}")

            setup = VehicleSetup(
                arb_front=af, arb_rear=ar, wing_position=w,
                tyre_pressure=float(pressure), brake_bias=float(bias),
                setup_name=f"ARB{af}/{ar}_W{w}",
            )
            params = apply_setup(base, setup)
            params_dict = params.to_solver_dict()

            try:
                r = _cached_solver(
                    params_dict=params_dict, circuit=circuit,
                    config={"gear_min": 1}, save_csv=False,
                )
                results_opt.append({
                    "arb_f": af, "arb_r": ar, "wing": w,
                    "lap_time": r["lap_time"],
                    "vmax_kmh": float(np.max(r["v_profile"])) * 3.6,
                    "vmean_kmh": float(np.mean(r["v_profile"])) * 3.6,
                })
            except Exception:
                results_opt.append({
                    "arb_f": af, "arb_r": ar, "wing": w,
                    "lap_time": float("inf"),
                    "vmax_kmh": 0.0, "vmean_kmh": 0.0,
                })

        elapsed = time.perf_counter() - t0
        bar.empty()

        df_opt = pd.DataFrame(results_opt)
        best = df_opt.loc[df_opt["lap_time"].idxmin()]

        st.success(
            f"✅ Optimisation complete in {elapsed:.1f}s — "
            f"Best: **ARB {int(best.arb_f)}/{int(best.arb_r)} Wing {int(best.wing)}** "
            f"→ **{fmt_laptime(best.lap_time)}**"
        )

        # Top 10 table
        st.subheader("🏆 Top 10 Setups")
        top10 = df_opt.nsmallest(10, "lap_time").copy()
        top10["Lap Time"] = top10["lap_time"].apply(fmt_laptime)
        top10.columns = [c.replace("_", " ").title() for c in top10.columns]
        st.dataframe(top10, width='stretch')

        # Heatmaps — one per wing position
        st.subheader("🗺️ Lap-Time Heatmaps (ARB Front × ARB Rear)")
        for w in wing_vals:
            sub = df_opt[df_opt["wing"] == w]
            if sub.empty:
                continue
            pivot = sub.pivot(index="arb_r", columns="arb_f",
                              values="lap_time")
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale='RdYlGn_r',
                colorbar=dict(title='Lap (s)'),
                text=[[fmt_laptime(v) for v in row] for row in pivot.values],
                texttemplate="%{text}",
            ))
            fig_hm.update_layout(
                title=f'Wing = {w}',
                xaxis_title='ARB Front', yaxis_title='ARB Rear',
                height=350, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_hm, width='stretch')


# ---------------------------------------------------------------------------
# APP ENTRY POINT
# ---------------------------------------------------------------------------
PAGES = {
    "🚗 Parameters":    parametros_veiculo_page,
    "🗺️ Track":         pista_page,
    "▶️ Simulation":    simulacao_page,
    "🏁 Results":       resultados_page,
    "📊 Compare":       compare_page,
    "🔧 Optimization":  optimization_page,
}

st.set_page_config(page_title="LapTimeSimulator — Copa Truck / GT3",
                   layout="wide", page_icon="🏁")
init_session_state()

st.sidebar.title("🏁 LapTimeSimulator")
st.sidebar.caption("Copa Truck | Porsche GT3 Cup")
page = st.sidebar.radio("Navigate:", list(PAGES.keys()))
PAGES[page]()
