# src/interface.py (COMPLETO E OTIMIZADO)
import streamlit as st
import os
import plotly.graph_objects as go
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
from tracks.hdf5 import CircuitHDF5Reader, CircuitData
from simulation import run_bicycle_model

DATA_PATH = r"C:\Users\vitor\OneDrive\Desktop\Pastas\LapTimeSimulator_CopaTruck\data\tracks"
RESULTS_PATH = r"C:\Users\vitor\OneDrive\Desktop\Pastas\LapTimeSimulator_CopaTruck\results"
MODELS_PATH = "data/vehicle_models.json"
os.makedirs(RESULTS_PATH, exist_ok=True)

def load_vehicle_models():
    """Carrega modelos de veículos do JSON"""
    try:
        with open(MODELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Arquivo {MODELS_PATH} não encontrado!")
        return {}

VEHICLE_MODELS = load_vehicle_models()

def init_session_state():
    """Inicializa session_state"""
    if "params" not in st.session_state:
        st.session_state.params = {
            "m": 5000.0, "lf": 2.1, "lr": 2.3, "h_cg": 1.1,
            "Cf": 120000.0, "Cr": 120000.0, "mu": 1.1, "r_wheel": 0.65,
            "P_max": 600000.0, "T_max": 3700.0, "rpm_max": 2800.0, "rpm_idle": 800.0,
            "n_gears": 12, "gear_ratios": [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78],
            "final_drive": 5.33, "max_decel": 7.5, "Cx": 0.85, "A_front": 8.7, "Cl": 0.0
        }
    if "config" not in st.session_state:
        st.session_state.config = {"tipo": "Qualificatória", "coef_aderencia": 1.09, "consumo": 43.0, "temp_pneu_ini": 65.0}
    if "circuit" not in st.session_state:
        st.session_state.circuit = None
    if "circuit_meta" not in st.session_state:
        st.session_state.circuit_meta = None
    if "resultados_prontos" not in st.session_state:
        st.session_state.resultados_prontos = False
    if "resultados" not in st.session_state:
        st.session_state.resultados = None
    if "last_csv_path" not in st.session_state:
        st.session_state.last_csv_path = None

def parametros_veiculo_page():
    """Página de parâmetros do veículo"""
    st.header("Parâmetros do Veículo - Copa Truck (2DOF)")
    
    st.subheader("📦 Modelo Pré-configurado")
    model_names_list = [v["name"] for v in VEHICLE_MODELS.values()]
    selected_model_name = st.selectbox("Selecione:", model_names_list, key="model_select")
    
    selected_model_key = None
    for key, model in VEHICLE_MODELS.items():
        if model["name"] == selected_model_name:
            selected_model_key = key
            break
    
    if selected_model_key:
        model_data = VEHICLE_MODELS[selected_model_key]
        st.session_state.params.update({
            "m": model_data["m"], "lf": model_data["lf"], "lr": model_data["lr"], "h_cg": model_data["h_cg"],
            "Cf": model_data["Cf"], "Cr": model_data["Cr"], "mu": model_data["mu"], "r_wheel": model_data["r_wheel"],
            "P_max": model_data["P_max"], "T_max": model_data["T_max"], "rpm_max": model_data["rpm_max"],
            "rpm_idle": model_data.get("rpm_idle", 800.0), "n_gears": model_data["n_gears"],
            "gear_ratios": model_data.get("gear_ratios", [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78]),
            "final_drive": model_data["final_drive"], "max_decel": model_data["max_decel"],
            "Cx": model_data["Cx"], "A_front": model_data["A_front"], "Cl": model_data["Cl"]
        })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fabricante", model_data["manufacturer"])
        with col2:
            st.metric("Ano", model_data["year"])
        with col3:
            st.metric("Potência", f"{model_data['P_max']/1000:.0f} kW")
        
        st.success(f"✓ {model_data['name']} carregado!")
    
    st.markdown("---")
    
    aba = st.radio("Customizar?", ["Não", "Sim"], horizontal=True, key="custom_radio")
    
    if aba == "Sim":
        section = st.radio("Seção:", ["Massa/Geometria", "Pneu", "Motor", "Transmissão", "Freio", "Aerodinâmica"], horizontal=True)
        
        if section == "Massa/Geometria":
            st.session_state.params["m"] = st.number_input("Massa (kg)", 3500.0, 9000.0, st.session_state.params["m"])
            entre_eixos = st.number_input("Entre-eixos (m)", 3.8, 5.5, st.session_state.params["lf"] + st.session_state.params["lr"])
            st.session_state.params["lf"] = st.number_input("Dist. CG → dianteira (m)", 1.0, 3.0, st.session_state.params["lf"])
            st.session_state.params["lr"] = entre_eixos - st.session_state.params["lf"]
            st.session_state.params["h_cg"] = st.number_input("Altura CG (m)", 0.7, 1.5, st.session_state.params["h_cg"])
        
        elif section == "Pneu":
            st.session_state.params["Cf"] = st.number_input("Cf dianteira (N/rad)", 60000.0, 250000.0, st.session_state.params["Cf"])
            st.session_state.params["Cr"] = st.number_input("Cr traseira (N/rad)", 60000.0, 250000.0, st.session_state.params["Cr"])
            st.session_state.params["mu"] = st.number_input("μ", 0.8, 1.5, st.session_state.params["mu"])
            st.session_state.params["r_wheel"] = st.number_input("Raio (m)", 0.5, 1.25, st.session_state.params["r_wheel"])
        
        elif section == "Motor":
            st.session_state.params["P_max"] = st.number_input("Potência (kW)", 300.0, 900.0, st.session_state.params["P_max"]/1000.0) * 1000.0
            st.session_state.params["T_max"] = st.number_input("Torque (Nm)", 2000.0, 6500.0, st.session_state.params["T_max"])
            st.session_state.params["rpm_max"] = st.number_input("RPM máx", 1800.0, 3500.0, st.session_state.params["rpm_max"])
            st.session_state.params["rpm_idle"] = st.number_input("RPM mín", 600.0, 1000.0, st.session_state.params.get("rpm_idle", 800.0))
        
        elif section == "Transmissão":
            st.session_state.params["n_gears"] = st.slider("Marchas", 6, 16, st.session_state.params["n_gears"])
            st.session_state.params["final_drive"] = st.number_input("Final drive", 2.8, 7.5, st.session_state.params["final_drive"])
        
        elif section == "Freio":
            st.session_state.params["max_decel"] = st.slider("Decel máx (m/s²)", 3.0, 10.0, st.session_state.params["max_decel"])
        
        elif section == "Aerodinâmica":
            st.session_state.params["Cx"] = st.number_input("Cx", 0.45, 1.20, st.session_state.params["Cx"])
            st.session_state.params["A_front"] = st.number_input("Área (m²)", 7.0, 10.0, st.session_state.params["A_front"])
            st.session_state.params["Cl"] = st.number_input("Cl", -1.0, 1.0, st.session_state.params["Cl"])

def pista_page():
    """Página de seleção de pista"""
    st.header("Pista")
    if not os.path.isdir(DATA_PATH):
        st.warning(f"Pasta não encontrada: {DATA_PATH}")
        return
    
    pistas = [f for f in os.listdir(DATA_PATH) if f.endswith('.hdf5')]
    if not pistas:
        st.warning('Nenhuma pista encontrada!')
        return
    
    pista_selecionada = st.selectbox("Selecione:", pistas)
    caminho_pista = os.path.join(DATA_PATH, pista_selecionada)
    circuit, meta = CircuitHDF5Reader(caminho_pista).read_circuit()
    
    st.session_state.circuit = circuit
    st.session_state.circuit_meta = meta
    
    x_c = -(circuit.centerline_y - circuit.centerline_y[0])
    y_c = circuit.centerline_x - circuit.centerline_x[0]
    left_x = -(circuit.left_boundary_y - circuit.centerline_y[0])
    left_y = circuit.left_boundary_x - circuit.centerline_x[0]
    right_x = -(circuit.right_boundary_y - circuit.centerline_y[0])
    right_y = circuit.right_boundary_x - circuit.centerline_x[0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_c, y=y_c, mode="lines", name="Centerline", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=left_x, y=left_y, mode="lines", name="Left", line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=right_x, y=right_y, mode="lines", name="Right", line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=[x_c[0]], y=[y_c[0]], mode="markers", marker=dict(color="orange", size=16, symbol="x"), name="Start"))
    fig.update_layout(title=f"{meta['name']}", xaxis_title="x (m)", yaxis_title="y (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"✓ {meta['name']} | {meta['length']:.2f}m")

def simulacao_page():
    """Página de configuração e execução da simulação"""
    st.header("Configuração da Simulação")
    
    if st.session_state.circuit is None:
        st.warning("⚠️ Selecione uma pista primeiro!")
        return
    
    st.info(f"✓ Pista: {st.session_state.circuit_meta['name']}")
    
    st.session_state.config["tipo"] = st.selectbox("Tipo:", [
        "Qualificatória", "Treino Livre", "Stint Longo", "Ultrapassagem", "Largada", "Aquecimento Pneus"
    ])
    st.session_state.config["coef_aderencia"] = st.slider("μ", 0.7, 1.4, st.session_state.config["coef_aderencia"])
    st.session_state.config["consumo"] = st.number_input("Consumo (l/100km)", 20.0, 70.0, st.session_state.config["consumo"])
    st.session_state.config["temp_pneu_ini"] = st.slider("Temp pneu (°C)", 30.0, 120.0, st.session_state.config["temp_pneu_ini"])
    
    col_play, col_reset = st.columns(2)
    
    with col_play:
        if st.button("▶ Play (Simular)", use_container_width=True):
            with st.spinner("🔄 Simulando..."):
                params_dict = {
                    "m": st.session_state.params["m"], "lf": st.session_state.params["lf"], "lr": st.session_state.params["lr"],
                    "h_cg": st.session_state.params["h_cg"], "Cf": st.session_state.params["Cf"], "Cr": st.session_state.params["Cr"],
                    "mu": st.session_state.config["coef_aderencia"], "r_wheel": st.session_state.params["r_wheel"],
                    "P_max": st.session_state.params["P_max"], "T_max": st.session_state.params["T_max"],
                    "rpm_max": st.session_state.params["rpm_max"], "rpm_idle": st.session_state.params.get("rpm_idle", 800.0),
                    "n_gears": st.session_state.params["n_gears"], "gear_ratios": st.session_state.params.get("gear_ratios", [14.0, 10.5, 7.8, 5.9, 4.5, 3.5, 2.7, 2.1, 1.6, 1.25, 1.0, 0.78]),
                    "final_drive": st.session_state.params["final_drive"], "max_decel": st.session_state.params["max_decel"],
                    "Cx": st.session_state.params["Cx"], "A_front": st.session_state.params["A_front"], "Cl": st.session_state.params["Cl"]
                }
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pista_nome = st.session_state.circuit_meta["name"].replace(" ", "_")
                csv_path = os.path.join(RESULTS_PATH, f"lap_{pista_nome}_{timestamp}.csv")
                
                resultados = run_bicycle_model(params_dict, st.session_state.circuit, st.session_state.config, save_csv=True, out_path=csv_path)
                
                st.session_state.resultados = resultados
                st.session_state.resultados_prontos = True
                st.session_state.last_csv_path = csv_path
                st.success(f"✓ Tempo: **{resultados['lap_time']:.2f}s**")
    
    with col_reset:
        if st.button("🔄 Reset Simulation", use_container_width=True):
            st.session_state.circuit = None
            st.session_state.circuit_meta = None
            st.session_state.resultados_prontos = False
            st.session_state.resultados = None
            st.session_state.last_csv_path = None
            st.info("✓ Reset OK. Selecione pista novamente.")
            st.rerun()

def resultados_page():
    """Página de resultados com tabela de KPIs"""
    st.header("Resultados")
    
    if not st.session_state.get("resultados_prontos", False):
        st.warning("Execute uma simulação primeiro!")
        return
    
    res = st.session_state.resultados
    
    # ===== KPIs =====
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Tempo", f"{res['lap_time']:.2f}s")
    with col2:
        v_max = np.max(res['v_profile']) * 3.6
        st.metric("V. Max", f"{v_max:.1f} km/h")
    with col3:
        v_avg = np.mean(res['v_profile']) * 3.6
        st.metric("V. Avg", f"{v_avg:.1f} km/h")
    with col4:
        a_lat_max = np.max(np.abs(res['a_lat']))
        st.metric("a_lat Max", f"{a_lat_max:.2f} m/s²")
    with col5:
        a_long_max = np.max(np.abs(res['a_long']))
        st.metric("a_long Max", f"{a_long_max:.2f} m/s²")
    
    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        t_max = np.max(res.get('temp_pneu', [70]))
        st.metric("T Pneu", f"{t_max:.1f}°C")
    with col7:
        rpm_max = np.max(res['rpm']) if len(res['rpm']) > 0 else 0
        st.metric("RPM Max", f"{rpm_max:.0f}")
    with col8:
        cons = res['consumo'][-1] if len(res['consumo']) > 0 else 0
        st.metric("Consumo", f"{cons:.2f}L")
    with col9:
        gear_max = int(np.max(res['gear']))
        st.metric("Marcha Max", gear_max)
    with col10:
        a_rms = np.sqrt(np.mean(res['a_long']**2 + res['a_lat']**2))
        st.metric("a_RMS", f"{a_rms:.2f} m/s²")
    
    st.markdown("---")
    
    # ===== TABELA DE TELEMETRIA =====
    st.subheader("📊 Telemetria Detalhada")
    
    telemetry_data = {
        "Distance (m)": np.round(res['distance'], 1),
        "Time (s)": np.round(res['time'], 2),
        "Velocity (km/h)": np.round(res['v_profile'] * 3.6, 1),
        "a_long (m/s²)": np.round(res['a_long'], 2),
        "a_lat (m/s²)": np.round(res['a_lat'], 2),
        "Gear": res['gear'].astype(int),
        "RPM": np.round(res['rpm'], 0).astype(int),
        "Radius (m)": np.round(res['radius'], 0).astype(int),
        "Temp (°C)": np.round(res.get('temp_pneu', [70]*len(res['v_profile'])), 1),
        "Consumo (L)": np.round(res['consumo'], 3),
    }
    
    telemetry_df = pd.DataFrame(telemetry_data)
    
    with st.expander("Ver tabela completa (downsample 1 a cada 5 pontos)"):
        st.dataframe(telemetry_df.iloc[::5], use_container_width=True, height=400)
    
    st.markdown("---")
    
    # ===== GRÁFICOS =====
    st.subheader("📈 Gráficos")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=res['distance'], y=res['v_profile']*3.6, mode="lines", name="Velocidade", line=dict(color="blue", width=2)))
        fig_v.update_layout(title="Velocidade", xaxis_title="Distância (m)", yaxis_title="km/h", height=350)
        st.plotly_chart(fig_v, use_container_width=True)
    
    with col_g2:
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=res['distance'], y=res['a_lat'], mode="lines", name="a_lateral", line=dict(color="red", width=2)))
        fig_a.update_layout(title="Aceleração Lateral", xaxis_title="Distância (m)", yaxis_title="m/s²", height=350)
        st.plotly_chart(fig_a, use_container_width=True)
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        fig_gear = go.Figure()
        fig_gear.add_trace(go.Scatter(x=res['distance'], y=res['gear'], mode="lines", name="Marcha", line=dict(color="green", width=2)))
        if 'rpm' in res and len(res['rpm']) > 0:
            fig_gear.add_trace(go.Scatter(x=res['distance'], y=np.array(res['rpm'])/200, mode="lines", name="RPM (÷200)", line=dict(color="orange", width=1, dash="dot")))
        fig_gear.update_layout(title="Marcha & RPM", xaxis_title="Distância (m)", yaxis_title="Valor", height=350)
        st.plotly_chart(fig_gear, use_container_width=True)
    
    with col_g4:
        if 'temp_pneu' in res:
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=res['distance'], y=res['temp_pneu'], mode="lines", name="Temp", line=dict(color="purple", width=2)))
            fig_temp.update_layout(title="Temperatura Pneu", xaxis_title="Distância (m)", yaxis_title="°C", height=350)
            st.plotly_chart(fig_temp, use_container_width=True)
    
    st.markdown("---")
    
    # ===== DOWNLOAD =====
    if st.session_state.get("last_csv_path") and os.path.exists(st.session_state["last_csv_path"]):
        with open(st.session_state["last_csv_path"], "rb") as f:
            st.download_button("📥 Download CSV", f, f"lap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

PAGES = {
    "Parâmetros": parametros_veiculo_page,
    "Pista": pista_page,
    "Simulação": simulacao_page,
    "Resultados": resultados_page
}

# ===== MAIN =====
st.set_page_config(page_title="LapTimeSimulator", layout="wide")
init_session_state()

st.sidebar.title("🏁 LapTimeSimulator")
page = st.sidebar.radio("Escolha:", list(PAGES.keys()))
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

with st.sidebar.expander("📋 Parâmetros"):
    st.write(f"**Massa:** {st.session_state.params['m']:.0f} kg")
    st.write(f"**Potência:** {st.session_state.params['P_max']/1000.0:.0f} kW")
    st.write(f"**μ:** {st.session_state.config['coef_aderencia']:.2f}")
    if st.session_state.circuit_meta:
        st.write(f"**Pista:** {st.session_state.circuit_meta['name']}")

PAGES[page]()
