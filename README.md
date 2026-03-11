# LapTimeSimulator_StockCar 🏁

> Simulador de tempo de volta de alta fidelidade para a **BRB Stock Car Pro Series — Temporada 2026**.

Projeto em Python modular para simulação dinâmica veicular, análise de telemetria (MoTec), otimização de setup, análise de pilotagem e gestão completa de finais de semana de corrida.

---

## 🗂️ Estrutura do Projeto

```
LapTimeSimulator_StockCar/
├── config/
│   ├── vehicles/            # Parâmetros dos veículos SNG01 (.yaml)
│   └── tracks/              # Dados das 12 pistas do calendário 2026 (.yaml)
├── core/
│   ├── vehicle_model/       # Modelo dinâmico parametrizado (2–14 DOF)
│   ├── track_model/         # Discretização e representação das pistas
│   └── lap_time_sim/        # Engine de simulação quasi-estática
├── data_pipeline/
│   ├── motec_importer/      # Parser de .csv/.ld MoTec → DataFrame
│   ├── channels/            # Canais matemáticos derivados
│   └── data_analysis/       # Análise comparativa de sessões
├── setup_optimizer/
│   ├── parameter_sweep/     # Varredura paramétrica de setup
│   └── optimizer/           # Otimização via SciPy/Optuna
├── driving_analysis/
│   ├── braking_efficiency/  # Índice de eficiência de frenagem por curva
│   ├── throttle_application/# Análise de aplicação de acelerador
│   └── lap_overlay/         # Comparativo lap-to-lap e piloto-a-piloto
├── weekend_manager/
│   ├── session_logger/      # Log estruturado por sessão
│   ├── tire_management/     # Controle de sets de pneus
│   └── endurance_mode/      # Módulo etapa Goiânia 3h (troca de pilotos)
├── kpis/
│   └── kpi_engine.py        # Cálculo e exportação de KPIs
├── reports/
│   ├── pre_event/           # Relatório pré-etapa
│   └── post_event/          # Relatório pós-etapa + exportação PDF/CSV
├── tests/                   # Testes unitários (pytest)
├── data/                    # Dados brutos e processados (gitignored)
├── docs/                    # Documentação técnica
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🏎️ Veículos Suportados (Geração SNG01)

| Modelo | Fabricante | Peso | Câmbio |
|--------|-----------|------|--------|
| Tracker | Chevrolet | ~1100 kg | XTrac P1529 6M |
| Corolla Cross | Toyota | ~1100 kg | XTrac P1529 6M |
| Eclipse Cross | Mitsubishi | ~1100 kg | XTrac P1529 6M |

## 📅 Calendário 2026 (12 Etapas)

| # | Data | Local | Circuito |
|---|------|-------|----------|
| 01 | 08/03 | Curvelo/MG | Circuito dos Cristais |
| 02 | 29/03 | Santa Cruz do Sul/RS | Autódromo Internacional SCS |
| 03 | 26/04 | São Paulo/SP | Autódromo de Interlagos |
| 04 | 17/05 | Goiânia/GO | Autódromo Int. Ayrton Senna |
| 05 | 13/06 | Cuiabá/MT | Autódromo Internacional MT |
| 06 | 26/07 | Mogi Guaçu/SP | Autódromo Velocitta |
| 07 | 09/08 | Cascavel/PR | Autódromo Zilmar Beux |
| 08 | 06/09 | Chapecó/SC | Autódromo Int. de Chapecó |
| 09 | 27/09 | Brasília/DF | Autódromo Int. Nelson Piquet |
| 10 | 18/10 | Goiânia/GO | Autódromo Int. Ayrton Senna *(Endurance 3h)* |
| 11 | 11/11 | Novo Hamburgo/RS | Velopark |
| 12 | 13/12 | São Paulo/SP | Autódromo de Interlagos *(Super Final)* |

## 🛠️ Instalação

```bash
git clone https://github.com/vitormtt/LapTimeSimulator_StockCar.git
cd LapTimeSimulator_StockCar
pip install -r requirements.txt
```

## 📄 Licença
MIT
