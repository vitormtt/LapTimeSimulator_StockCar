# LapTimeSimulator_CopaTruck 🏁🚛

Simulador de Tempo de Volta (Lap Time Simulator) desenvolvido para veículos de competição do tipo **Copa Truck**, com foco em análise de desempenho dinâmico em ambientes virtuais controlados.

*Projeto vinculado ao Mestrado em Engenharia Automotiva (UnB) - Foco em Dinâmica Veicular e Simulação Computacional.*

## 🛠️ Arquitetura e Física do Modelo
O simulador utiliza uma abordagem **Quasi-Steady-State (QSS)** baseada na integração Forward-Backward e no modelo de **Bicicleta (2-DOF)**, expandido com cálculos dinâmicos avançados:

* **Diagrama GGV Dinâmico Acoplado**: Cálculo de limites de aderência (Friction Circle) considerando *Downforce* e Arrasto Aerodinâmico atuando iterativamente na Força Normal ($F_z$).
* **Transferência de Carga Longitudinal**: Modulação dinâmica de *Pitch* (*Squat* em aceleração e *Dive* em frenagem) alterando a aderência disponível nos eixos traseiro e dianteiro.
* **Powertrain Modular**: Modelagem realista de Curvas de Torque para Motor Diesel 12L, relações de transmissão limitadas a parâmetros de corrida (rolling start/4ª marcha), e mapa de consumo de combustível termodinâmico.
* **Frenagem Pneumática**: Limitadores mecânicos baseados em câmaras de ar reais e distribuição de freio (Brake Balance 60/40).
* **Tratamento de Trajetória**: Suavização de malha via filtro *Savitzky-Golay* para mitigação de picos derivativos em pistas geradas por dados de GPS/OSM reais.

## 🚀 Instalação e Execução

### 1. Clonar o repositório
```powershell
git clone https://github.com/vitormtt/LapTimeSimulator_CopaTruck.git
cd LapTimeSimulator_CopaTruck
```

### 2. Sincronizar as Dependências
O projeto conta com um ecossistema padronizado de Data Science e UI. Para instalar ou atualizar todas as bibliotecas necessárias, execute:
```powershell
pip install --upgrade -r requirements.txt
```

### 3. Pistas (Opcional)
Na ausência de malhas GPS/OSM brutas mapeadas na sua máquina, você pode gerar pistas sintéticas dimensionadas para validar o motor do simulador:
```powershell
python src/tracks/generate_br_tracks.py
```

### 4. Iniciar a Interface
Para abrir o Dashboard interativo (Streamlit):
```powershell
streamlit run src/visualization/interface.py
```

## 📚 Referências Acadêmicas Base
- **Casanova, D. (2000)** - *On minimum time vehicle manoeuvring: The theoretical optimal lap.* (Fundamentação algorítmica para a solução QSS Forward-Backward).
- **Gillespie, T.D. (1992)** - *Fundamentals of Vehicle Dynamics.* (Equações de movimento, taxas de Yaw e transferência de carga elástica).
- **Pacejka, H.B. (2012)** - *Tire and Vehicle Dynamics.* (Friction Circle e "Magic Formula" simplificada para estresse de banda de rodagem).
- **Savaresi, S. M. et al. (2010)** - *Automotive Semi-Active Suspensions.* (Embasamento da simplificação em 2-DOF macroscópica para o tempo de volta contínuo).

---
*Desenvolvido seguindo as diretrizes PEP 8 (Python) aplicadas à Engenharia Automotiva.*
