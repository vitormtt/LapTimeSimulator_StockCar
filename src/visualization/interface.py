import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
    
from src.tracks.hdf5 import CircuitHDF5Reader
from src.simulation.lap_time_solver import run_bicycle_model
from src.vehicle.parameters import copa_truck_2dof_default

DATA_PATH = str(BASE_DIR / "tracks")
RESULTS_PATH = str(BASE_DIR / "src" / "results")
os.makedirs(RESULTS_PATH, exist_ok=True)

@st.cache_data
def load_track_data(caminho_pista: str):
    t0 = time.time()
    circuit, meta = CircuitHDF5Reader(caminho_pista).read_circuit()
    t1 = time.time()
    logger.info(f"[PERFORMANCE] Pista {meta['name']} carregada da memória HDF5 em {t1 - t0:.4f}s")
    
    x_c = -(circuit.centerline_y - circuit.centerline_y[0])
    y_c = circuit.centerline_x - circuit.centerline_x[0]
    left_x = -(circuit.left_boundary_y - circuit.centerline_y[0])
    left_y = circuit.left_boundary_x - circuit.centerline_x[0]
    right_x = -(circuit.right_boundary_y - circuit.centerline_y[0])
    right_y = circuit.right_boundary_x - circuit.centerline_x[0]
    
    plot_data = {
        'x_c': x_c, 'y_c': y_c,
        'left_x': left_x, 'left_y': left_y,
        'right_x': right_x, 'right_y': right_y
    }
    
    return circuit, meta, plot_data

def init_session_state():
    if "vehicle_params" not in st.session_state:
        st.session_state.vehicle_params = copa_truck_2dof_default()
    if "config" not in st.session_state:
        st.session_state.config = {
            "tipo": "Qualificatória",
            "coef_aderencia": 1.09,
            "consumo": 43.0,
            "temp_pneu_ini": 65.0
        }
    if "circuit" not in st.session_state:
        st.session_state.circuit = None
    if "circuit_meta" not in st.session_state:
        st.session_state.circuit_meta = None
    if "resultados_prontos" not in st.session_state:
        st.session_state.resultados_prontos = False
    if "resultados" not in st.session_state:
        st.session_state.resultados = None
    if "csv_path" not in st.session_state:
        st.session_state.csv_path = None

def parametros_veiculo_page():
    st.header("Vehicle Parameters - Copa Truck (3-DOF)")
    vp = st.session_state.vehicle_params

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Manufacturer", vp.manufacturer)
    with col2: st.metric("Year", vp.year)
    with col3: st.metric("Power", f"{vp.engine.max_power/1000:.0f} kW")

    st.markdown("---")
    aba = st.radio("Customize?", ["No", "Yes"], horizontal=True, key="custom_radio")

    if aba == "Yes":
        section = st.radio("Section:", ["Mass/Geometry", "Tire", "Engine", "Transmission", "Brake", "Aerodynamics"], horizontal=True)

        if section == "Mass/Geometry":
            vp.mass_geometry.mass = st.number_input("Mass (kg)", 3500.0, 9000.0, float(vp.mass_geometry.mass))
            wheelbase = st.number_input("Wheelbase (m)", 3.8, 5.5, float(vp.mass_geometry.wheelbase))
            vp.mass_geometry.lf = st.number_input("Dist. CG → front (m)", 1.0, 3.0, float(vp.mass_geometry.lf))
            vp.mass_geometry.lr = wheelbase - vp.mass_geometry.lf
            vp.mass_geometry.wheelbase = wheelbase
            st.markdown("### 3-DOF Roll Parameters")
            st.number_input("K_roll (Nm/rad)", value=450000.0, step=50000.0, help="Rigidez torcional da barra estabilizadora e molas combinadas")
            st.number_input("Track Width (m)", value=2.45, help="Bitola do eixo")
        elif section == "Tire":
            vp.tire.cornering_stiffness_front = st.number_input("Cf front (N/rad)", 60000.0, 250000.0, float(vp.tire.cornering_stiffness_front))
            vp.tire.cornering_stiffness_rear = st.number_input("Cr rear (N/rad)", 60000.0, 250000.0, float(vp.tire.cornering_stiffness_rear))
            vp.tire.friction_coefficient = st.number_input("μ (base)", 0.8, 1.5, float(vp.tire.friction_coefficient))
        elif section == "Engine":
            vp.engine.max_power = st.number_input("Power (kW)", 300.0, 900.0, float(vp.engine.max_power)/1000.0) * 1000.0
            vp.engine.max_torque = st.number_input("Torque (Nm)", 2000.0, 6500.0, float(vp.engine.max_torque))
        elif section == "Transmission":
            vp.transmission.num_gears = st.slider("Gears", 6, 16, int(vp.transmission.num_gears))
            vp.transmission.final_drive_ratio = st.number_input("Final drive", 2.8, 7.5, float(vp.transmission.final_drive_ratio))
        elif section == "Brake":
            vp.brake.max_deceleration = st.slider("Max decel (m/s²)", 3.0, 10.0, float(vp.brake.max_deceleration))
        elif section == "Aerodynamics":
            vp.aero.drag_coefficient = st.number_input("Cd", 0.45, 1.20, float(vp.aero.drag_coefficient))
            vp.aero.frontal_area = st.number_input("Frontal area (m²)", 7.0, 10.0, float(vp.aero.frontal_area))

def pista_page():
    st.header("Track")
    if not os.path.isdir(DATA_PATH):
        st.warning(f"Pasta de pistas não encontrada no caminho absoluto: {DATA_PATH}. Certifique-se de estar rodando na raiz do projeto.")
        return

    pistas = [f for f in os.listdir(DATA_PATH) if f.endswith('.hdf5')]
    if not pistas:
        st.warning(f'Nenhuma pista .hdf5 encontrada na pasta {DATA_PATH}!')
        return

    pista_selecionada = st.selectbox("Select:", pistas)
    caminho_pista = os.path.join(DATA_PATH, pista_selecionada)
    
    circuit, meta, plot_data = load_track_data(caminho_pista)

    st.session_state.circuit = circuit
    st.session_state.circuit_meta = meta

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=plot_data['x_c'], y=plot_data['y_c'], mode="lines", name="Centerline", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scattergl(x=plot_data['left_x'], y=plot_data['left_y'], mode="lines", name="Left", line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scattergl(x=plot_data['right_x'], y=plot_data['right_y'], mode="lines", name="Right", line=dict(color='red', dash='dot')))
    fig.update_layout(title=f"{meta['name']}", xaxis_title="x (m)", yaxis_title="y (m)", margin=dict(l=0, r=0, t=30, b=0))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig)

    st.success(f"✓ Pista carregada: {meta['name']} | {meta['length']:.2f}m")

def simulacao_page():
    st.header("Simulation Configuration")
    if st.session_state.circuit is None:
        st.warning("⚠️ Você precisa ir na aba 'Track' e selecionar uma pista primeiro antes de simular!")
        return

    st.info(f"✓ Track Pronta para Simulação: {st.session_state.circuit_meta['name']}")

    col_play, col_reset = st.columns(2)
    with col_play:
        if st.button("▶ Play (Simulate)", use_container_width=True):
            with st.spinner("🔄 Simulating..."):
                params_dict = st.session_state.vehicle_params.to_solver_dict()
                params_dict["mu"] = st.session_state.config["coef_aderencia"]
                # Injeta parâmetros básicos de Roll caso o usuário não tenha mexido
                params_dict["k_roll"] = 450000.0
                params_dict["track_width"] = 2.45

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pista_nome = st.session_state.circuit_meta["name"].replace(" ", "_")
                csv_path = os.path.join(RESULTS_PATH, f"lap_{pista_nome}_{timestamp}.csv")

                resultados = run_bicycle_model(params_dict, st.session_state.circuit, st.session_state.config, save_csv=True, out_path=csv_path)

                st.session_state.resultados = resultados
                st.session_state.csv_path = csv_path
                st.session_state.resultados_prontos = True
                st.success(f"✓ Lap time: **{resultados['lap_time']:.2f}s** (Computed in {resultados.get('compute_time_s', 0):.3f}s)")

def resultados_page():
    st.header("Results & Telemetry")
    if not st.session_state.get("resultados_prontos", False):
        st.warning("⚠️ Execute a simulação na aba 'Simulation' primeiro para ver os resultados aqui!")
        return

    res = st.session_state.resultados
    csv_file = st.session_state.csv_path

    # Advanced KPIs (Math Channels)
    # Extrai do dicionário para fácil cálculo
    v_kmh = res['v_profile'] * 3.6
    a_long_g = res['a_long'] / 9.81
    a_lat_g = res['a_lat'] / 9.81
    
    time_w_full_throttle = np.sum(np.where(a_long_g > 0.1, 1, 0)) / len(a_long_g) * 100
    time_braking = np.sum(np.where(a_long_g < -0.2, 1, 0)) / len(a_long_g) * 100
    time_coasting = np.sum(np.where((a_long_g >= -0.2) & (a_long_g <= 0.1), 1, 0)) / len(a_long_g) * 100
    
    max_lat_g = np.max(np.abs(a_lat_g))
    max_braking_g = np.min(a_long_g)
    max_roll_angle = np.max(np.abs(res['roll_angle_profile'])) if 'roll_angle_profile' in res else 0.0

    st.subheader("🏁 Performance KPIs")
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric("Lap Time", f"{res['lap_time']:.2f} s")
        st.metric("Max Speed", f"{np.max(v_kmh):.1f} km/h")
        st.metric("Avg Speed", f"{np.mean(v_kmh):.1f} km/h")
    with col2: 
        st.metric("Time @ WOT", f"{time_w_full_throttle:.1f} %", help="Wide Open Throttle")
        st.metric("Time Braking", f"{time_braking:.1f} %")
        st.metric("Time Coasting", f"{time_coasting:.1f} %")
    with col3: 
        st.metric("Peak Cornering G", f"{max_lat_g:.2f} G")
        st.metric("Peak Braking G", f"{abs(max_braking_g):.2f} G")
        st.metric("Peak Accel G", f"{np.max(a_long_g):.2f} G")
    with col4: 
        st.metric("Fuel Used", f"{np.max(res['consumo']):.2f} L")
        st.metric("Peak Cabin Roll", f"{max_roll_angle:.2f} °")
        st.metric("Gear Shifts", f"{np.sum(np.abs(np.diff(res['gear'])))}", help="Total de trocas de marcha na volta")
    
    st.markdown("---")
    
    # Download Button Formatado
    if csv_file and os.path.exists(csv_file):
        with open(csv_file, "rb") as f:
            st.download_button(
                label="📥 Baixar Telemetria Completa (.CSV) para PiToolbox/MoTeC",
                data=f,
                file_name=f"CopaTruck_Sim_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                help="Este arquivo contém todos os Math Channels (G_Lat, G_Long, Speed, Yaw, Slip_Angle, etc.) padronizados para softwares de engenharia."
            )
            
    st.markdown("---")
    st.subheader("📈 Telemetry Charts")

    # Primeira Linha
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scattergl(x=res['distance'], y=v_kmh, mode="lines", name="Speed", line=dict(color="blue", width=2)))
        fig_v.update_layout(title="Speed (km/h)", margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_v)

    with col_g2:
        fig_a = go.Figure()
        fig_a.add_trace(go.Scattergl(x=res['distance'], y=a_lat_g, mode="lines", name="Lat G", line=dict(color="red", width=2)))
        fig_a.update_layout(title="Lateral Accel (G)", margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_a)
        
    # Segunda Linha
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        fig_long = go.Figure()
        fig_long.add_trace(go.Scattergl(x=res['distance'], y=a_long_g, mode="lines", name="Long G", line=dict(color="orange", width=2)))
        fig_long.update_layout(title="Longitudinal Accel (G)", margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_long)
        
    with col_g4:
        if 'roll_angle_profile' in res:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scattergl(x=res['distance'], y=res['roll_angle_profile'], mode="lines", name="Roll Angle", line=dict(color="purple", width=2)))
            fig_roll.update_layout(title="Cabin Roll Angle (degrees) [3-DOF]", margin=dict(l=0, r=0, t=30, b=0), height=300)
            st.plotly_chart(fig_roll)
        else:
            st.info("Dado de Roll não disponível.")
            
    # Terceira Linha - RPM e Marcha (GGV Combinado)
    st.markdown("---")
    st.subheader("⚙️ Powertrain & GGV")
    col_g5, col_g6 = st.columns(2)
    with col_g5:
        fig_rpm = go.Figure()
        fig_rpm.add_trace(go.Scattergl(x=res['distance'], y=res['rpm'], mode="lines", name="RPM", line=dict(color="green", width=2)))
        fig_rpm.add_trace(go.Scattergl(x=res['distance'], y=res['gear']*200, mode="lines", name="Gear (*200)", line=dict(color="grey", width=1, dash='dash')))
        fig_rpm.update_layout(title="Engine RPM & Gear", margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_rpm)
        
    with col_g6:
        # GGV Diagram
        fig_ggv = go.Figure()
        fig_ggv.add_trace(go.Scatter(x=a_lat_g, y=a_long_g, mode='markers', marker=dict(size=4, color=v_kmh, colorscale='Viridis', showscale=True, colorbar=dict(title="Speed"))))
        fig_ggv.update_layout(title="GGV Diagram (Friction Circle)", xaxis_title="Lat G", yaxis_title="Long G", width=400, height=400, yaxis_range=[-1.2, 1.2], xaxis_range=[-1.2, 1.2])
        # Traçar o círculo teórico
        theta = np.linspace(0, 2*np.pi, 100)
        mu = st.session_state.config["coef_aderencia"]
        fig_ggv.add_trace(go.Scatter(x=mu*np.cos(theta), y=mu*np.sin(theta), mode='lines', line=dict(color='red', dash='dash'), name='Friction Limit'))
        st.plotly_chart(fig_ggv)

PAGES = {
    "Parameters": parametros_veiculo_page,
    "Track": pista_page,
    "Simulation": simulacao_page,
    "Results": resultados_page
}

st.set_page_config(page_title="LapTimeSimulator", layout="wide")
init_session_state()

st.sidebar.title("🏁 LapTimeSimulator")
page = st.sidebar.radio("Choose:", list(PAGES.keys()))
PAGES[page]()
