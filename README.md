# 🏁 LapTimeSimulator\_CopaTruck

> Simulador de lap time e ferramenta de análise de dados para Copa Truck, baseado em modelo de dinâmica veicular (Bicycle Model 2DOF), interface web Streamlit e arquitetura inspirada no workflow do **Pi Toolbox / Cosworth**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io/)
[![Branch](https://img.shields.io/badge/dev-feature%2Ffrontend--v2-orange)](https://github.com/vitormtt/LapTimeSimulator_CopaTruck/tree/feature/frontend-v2)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Visão Geral

A engenharia automotiva moderna exige ferramentas de simulação eficientes para análise do desempenho dinâmico em ambiente controlado. O LapTimeSimulator\_CopaTruck combina um solver físico de lap time com um painel de análise de dados interativo, cobrindo os mesmos tipos de visualização do **Pi Toolbox** (Cosworth): gráficos tempo/distância, mapa da pista, diagrama GG (X-Y), histogramas, tabelas comparativas e canais matemáticos.

---

## 📊 Funcionalidades (v2 — branch `feature/frontend-v2`)

### Modelo Dinâmico
- **Bicycle Model 2DOF**: dinâmica longitudinal e lateral com círculo de aderência
- **Motor Diesel**: curva de torque calibrada para caminhões de corrida (pico ~1 300 RPM)
- **Transmissão multi-marcha**: até 16 marchas, seleção automática por faixa de RPM ótima
- **Aerodinâmica**: arrasto (Cd) e downforce (Cl) configuráveis
- **Modelo térmico simplificado de pneu**: temperatura e pressão ao longo da volta
- **Forward–Backward Pass**: aceleração máxima + frenagem por velocidade limite de curva

### Interface Web (Streamlit)

| Página | Descrição |
|---|---|
| ⚙️ **Parameters** | Configuração do veículo por abas: massa/geometria, pneu, motor, transmissão, freios, aerodinâmica |
| 🏎️ **Track** | Seleção e preview interativo do circuito (HDF5), com limites e ponto de S/F |
| ▶️ **Simulation** | Configuração de sessão + botão Simulate + **Save Lap** para gerenciamento de sessão |
| 📊 **Results** | KPIs (10 métricas), gráfico Velocidade vs Distância, **GG Diagram**, telemetria multi-canal selecionável, tabela raw com downsample |
| 🔄 **Compare Laps** | Tabela comparativa com Δ ao fastest, speed overlay, GG grid, channel overlay por canal |

### Análise de Dados (insp. Pi Toolbox / i2)

Cobertos os tipos de visualização documentados no **Treinamento Pi Toolbox – Porsche GT3 Cup Brasil**:

| Tipo Pi Toolbox | Implementação no Simulador |
|---|---|
| Gráfico distância/tempo | `plot_channels_vs_distance` — eixo X distância ou tempo |
| Mapa da pista | Página Track com Plotly Scatter + overlay de canais (futuro) |
| X-Y (GG diagram) | `plot_gg_diagram` — ax_long vs ay_lat, colorscale por velocidade |
| Histogramas | Planejado para v2.1 (distribuição de velocidades/G) |
| Tabelas comparativas | Página Compare Laps: min, méx, média, Δ ao fastest |
| Split Report / setores | Planejado para v2.1 (tabela de setores por volta) |
| Variance / Compare Time | Canal `delta_time` vs distância — planejado v2.1 |
| Canais matemáticos | `kpi_dashboard.py`: Slip Ratio, WOT%, Braking%, a_RMS, etc. |
| Vitals & Alarms | Linhas de referência nos gráficos (futuro: alertas automáticos) |

---

## 🚀 Início Rápido

### Instalação

```bash
# Clone o repositório
git clone https://github.com/vitormtt/LapTimeSimulator_CopaTruck.git
cd LapTimeSimulator_CopaTruck

# Crie o ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Instale as dependências
pip install -r requirements.txt
```

### Executar (branch de desenvolvimento)

```bash
git checkout feature/frontend-v2
streamlit run src/visualization/interface.py
```

Acesse no navegador: **http://localhost:8501**

---

## 📖 Como Usar

### 1. Selecionar Circuito 🏁
Navegue para **Track**, selecione um `.hdf5` da pasta `tracks/`, visualize o traçado e os limites.

### 2. Configurar Veículo ⚙️
Em **Parameters**, ajuste cada subsistema nas abas dedicadas. Os valores calculados (ex: distribuição de peso F/R) são exibidos em tempo real.

### 3. Simular ▶️
Em **Simulation**, defina tipo de sessão, μ de pista, consumo e temperatura inicial de pneu. Clique em ▶ Simulate. Use 💾 **Save Lap** para armazenar a volta na sessão com um label descritivo.

### 4. Analisar 📊
Em **Results**:
- **KPIs**: 10 métricas instantâneas
- **Speed + GG**: visão geral da volta
- **Multi-channel**: selecione os canais que deseja sobrepor (estilo *tiled* do Pi Toolbox)
- **Raw table**: downsample configurável + download CSV

### 5. Comparar Voltas 🔄
Em **Compare Laps**, selecione as voltas salvas na sessão:
- Tabela com Δ ao fastest, parâmetros de setup e KPIs
- Overlay de velocidade de todas as voltas em um único gráfico
- GG diagrams individuais em grid
- Channel overlay por canal selecionável

> **Dica**: Para comparar pilotos/setups como no Pi Toolbox, sempre use distância no eixo X — o canal de velocidade se expande nas retas e comprime nas curvas, facilitando a identificação dos pontos de freada e tangência.

---

## 📂 Estrutura do Projeto

```
LapTimeSimulator_CopaTruck/
├── src/
│   ├── simulation/          # Solver (lap_time_solver.py) + SimulationResult
│   ├── vehicle/             # VehicleParams (dataclasses) + presets
│   ├── tracks/              # Leitura/escrita HDF5, TUM FTM, gerador
│   ├── visualization/
│   │   ├── interface.py     # Streamlit app (5 páginas)
│   │   ├── kpi_dashboard.py # Funções Plotly: GG, speed overlay, channels, KPI table
│   │   └── track_plotter.py # Matplotlib fallback
│   ├── optimization/        # Otimização de setup (em desenvolvimento)
│   └── results/             # CSVs exportados automaticamente
├── tracks/                  # Circuitos .hdf5 (Interlagos, etc.)
├── data/                    # Modelos de veículos .json
├── requirements.txt
└── README.md
```

---

## 🔧 Arquitetura Técnica

### Solver (Forward–Backward Pass)
1. **Cálculo da velocidade limite por curva**: `v_max = sqrt(μ * g * R)` respeitando círculo de aderência
2. **Forward pass**: aceleração máxima limitada por força de tração e arrasto aerodinâmico
3. **Backward pass**: frenagem máxima limitada por desaceleração e círculo de aderência
4. **Integração temporal**: `Δt = Δs / v`, acumula tempo de volta por segmento

### Canais Matemáticos (`kpi_dashboard.py`)

| Canal | Expressão |
|---|---|
| `throttle_pct` | `clip(ax_long_g / max_ax * 100, 0, 100)` |
| `brake_pct` | `clip(-ax_long_g / max_ax * 100, 0, 100)` |
| `WOT (%)` | `mean(ax_long_g > 0.05) * 100` |
| `Braking (%)` | `mean(ax_long_g < -0.05) * 100` |
| `a_RMS` | `sqrt(mean(ax² + ay²))` |
| `Slip Ratio (futuro)` | `(V_roda - V_carro) / V_carro` |

### Formatos Suportados
- **Input circuito**: `.hdf5` (HDF5 comprimido com centerline + limites)
- **Input referência**: `.csv` externo (ex: dados 992.1 Porsche GT3 Cup) — importação na página Compare Laps (v2.1)
- **Output**: `.csv` com todos os canais exportado automaticamente após cada simulação

---

## 📊 Referência Externa: 992.1 Porsche GT3 Cup

O simulador suporta importação de CSVs de telemetria real (ex: dados exportados pelo Pi Toolbox / Cosworth do 992.1) para comparação direta com as voltas simuladas na página **Compare Laps**.

Colunas esperadas no CSV de referência:

```
distance_m, time_s, speed_kmh, ax_long_g, ay_lat_g, gear, rpm, throttle_pct, brake_pct, tyre_temp_c
```

O próximo passo é implementar o importador de CSV externo (`src/visualization/csv_importer.py`) e a normalização de canais para garantir compatibilidade com diferentes softwares de telemetria (Pi Toolbox, MoTeC i2, WinTax).

---

## 🎯 Roadmap

### v2.0 (branch `feature/frontend-v2` — atual)
- [x] Dark theme com CSS customizado
- [x] Navegação em 5 páginas com status card na sidebar
- [x] Parâmetros em abas por subsistema
- [x] GG Diagram com colorscale por velocidade
- [x] Multi-channel telemetry selecionável
- [x] Gestão de sessão multi-volta (Save Lap)
- [x] Compare Laps: tabela Δ, speed overlay, GG grid, channel overlay
- [x] Paths dinâmicos (sem hardcode de diretório)

### v2.1 (próximo)
- [ ] Importador de CSV externo (Pi Toolbox, MoTeC, WinTax) com normalização de canais
- [ ] Canal **Compare Time** (Δt cumulativo por distância) — equiv. ao *Variance/Compare Time* do Pi Toolbox
- [ ] **Split Report**: tabela de setores com Eclectic e Rolling Minimum
- [ ] Mapa da pista colorido por canal (velocidade, G-lat, marcha, sub/sobre-esterço)
- [ ] Histogramas de velocidade, G-lat e posição do acelerador
- [ ] Linhas de referência nos gráficos (limites de alarme configuráveis)
- [ ] `st.cache_data` no solver e leitura HDF5

### v3.0 (futuro)
- [ ] Modelo 3DOF com roll dynamics
- [ ] Otimização de setup automática (scipy.optimize / algoritmos genéticos)
- [ ] Modelo térmico de pneu Pacejka Magic Formula
- [ ] Estratégia de combustível e pit stop
- [ ] Export HTML de relatório completo (plotly.io.write_html)
- [ ] API REST para integração com telemetria real

---

## 🛠️ Desenvolvimento

### Adicionar Novo Circuito

```python
from src.tracks.hdf5 import CircuitHDF5Writer, CircuitData
import numpy as np

circuit = CircuitData(
    name="Autodóromo de Curitiba",
    centerline_x=np.array([...]),
    centerline_y=np.array([...]),
    left_boundary_x=left_x,
    left_boundary_y=left_y,
    right_boundary_x=right_x,
    right_boundary_y=right_y,
    track_width=track_width,
)
CircuitHDF5Writer("tracks/curitiba.hdf5").write_circuit(circuit)
```

### Criar Preset de Veículo

```python
from src.vehicle.parameters import VehicleParams, VehicleMassGeometry, TireParams

meu_caminhao = VehicleParams(
    mass_geometry=VehicleMassGeometry(mass=5500.0, lf=2.2, lr=2.4, ...),
    tire=TireParams(cornering_stiffness_front=130_000.0, ...),
    name="Volvo FH Custom",
    manufacturer="Volvo",
    year=2025,
)
meu_caminhao.save_to_json("data/volvo_fh_custom.json")
```

### Branches

| Branch | Status | Descrição |
|---|---|---|
| `main` | ✅ estável | Versão funcional com interface básica |
| `feature/frontend-v2` | 🚧 dev | Interface redesenhada, Compare Laps, GG diagram |

---

## 📚 Referências

- **Rajamani, R.** (2012) — *Vehicle Dynamics and Control*, Springer
- **Pacejka, H.** (2012) — *Tire and Vehicle Dynamics*, Butterworth-Heinemann
- **Gillespie, T.** (1992) — *Fundamentals of Vehicle Dynamics*, SAE International
- **Segers, J.** (2014) — *Analysis Techniques For Racecar Data Acquisition*, 2nd Ed., SAE International
- **Pi Toolbox / Cosworth** — Treinamento Porsche GT3 Cup Challenge Brasil (2014)
- **TUM FTM** — Racetrack Database: https://github.com/TUMFTM/racetrack-database

---

## 👨‍💻 Autor

**Vitor Mattos**
GitHub: [@vitormtt](https://github.com/vitormtt) · Universidade de Brasília — Mestrado em Engenharia Automotiva

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!
