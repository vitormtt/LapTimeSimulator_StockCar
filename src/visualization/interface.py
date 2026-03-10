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
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
import os
import sys
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

                # ARB k_roll from setup if GT3
                setup = st.session_state.get("setup")
                if setup is not None:
                    params_dict["k_roll"] = (
                        setup.arb_front_stiffness + setup.arb_rear_stiffness
                    )
                else:
                    params_dict.setdefault("k_roll", 450000.0)

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
                    result = run_bicycle_model(
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
# APP ENTRY POINT
# ---------------------------------------------------------------------------
PAGES = {
    "🚗 Parameters":   parametros_veiculo_page,
    "🗭️ Track":        pista_page,
    "▶️ Simulation":   simulacao_page,
    "🏁 Results":      resultados_page,
}

st.set_page_config(page_title="LapTimeSimulator — Copa Truck / GT3",
                   layout="wide", page_icon="🏁")
init_session_state()

st.sidebar.title("🏁 LapTimeSimulator")
st.sidebar.caption("Copa Truck | Porsche GT3 Cup")
page = st.sidebar.radio("Navigate:", list(PAGES.keys()))
PAGES[page]()
