"""
Gerador Estático e Deterministico de Pistas Reais
Substitui a dependência instável da API do OSM por interpolação paramétrica
baseada no traçado real dos autódromos.
"""
import numpy as np
import os
import sys
from pathlib import Path
from scipy.interpolate import splprep, splev

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tracks.circuit import CircuitData
from src.tracks.hdf5 import CircuitHDF5Writer

def create_track_from_waypoints(name: str, waypoints: list, num_points: int = 1500, track_width: float = 12.0) -> CircuitData:
    """
    Recebe pontos (X,Y) representativos de um autódromo e usa B-Splines 
    para gerar uma malha suave fechada altamente precisa para a simulação.
    """
    # Garante que o circuito fecha perfeitamente
    pts = np.array(waypoints)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack((pts, pts[0]))
        
    x, y = pts[:, 0], pts[:, 1]
    
    # Interpolação Spline Cúbica (k=3) periódica (s=0 significa ajuste perfeito nos pontos)
    tck, u = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1, num_points)
    out = splev(unew, tck)
    
    x_smooth, y_smooth = out[0], out[1]
    
    # Calcula Boundaries
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    
    # Normalização dos vetores para achar a perpendicular
    norms = np.sqrt(dx**2 + dy**2)
    dx_norm = dx / (norms + 1e-8)
    dy_norm = dy / (norms + 1e-8)
    
    # Vetor normal (rotação de 90 graus: -dy, dx)
    nx = -dy_norm
    ny = dx_norm
    
    left_x = x_smooth + nx * (track_width / 2.0)
    left_y = y_smooth + ny * (track_width / 2.0)
    right_x = x_smooth - nx * (track_width / 2.0)
    right_y = y_smooth - ny * (track_width / 2.0)
    
    # Vetor track_width
    width_array = np.full_like(x_smooth, track_width)
    
    circuit = CircuitData(
        name=name,
        centerline_x=x_smooth,
        centerline_y=y_smooth,
        left_boundary_x=left_x,
        left_boundary_y=left_y,
        right_boundary_x=right_x,
        right_boundary_y=right_y,
        track_width=width_array
    )
    
    return circuit

def build_offline_tracks():
    tracks_dir = ROOT_DIR / "tracks"
    os.makedirs(tracks_dir, exist_ok=True)
    
    # 1. INTERLAGOS REAL (Mapeamento aproximado em metros, Perímetro: ~4309m)
    # Sequência: Reta dos Boxes -> S do Senna -> Reta Oposta -> Descida do Lago -> Ferradura -> Pinheirinho -> Bico de Pato -> Mergulho -> Junção -> Subida dos Boxes
    interlagos_pts = [
        (0, 0), (200, 10), (450, 20), (550, -50), (450, -150), # Reta/Senna
        (350, -300), (200, -600), (100, -800), # Reta Oposta
        (0, -900), (-100, -850), (-150, -750), # Descida do Lago
        (-100, -650), (-50, -550), (-200, -500), # Ferradura
        (-350, -450), (-400, -350), (-250, -250), # Pinheirinho
        (-200, -100), (-300, 0), (-150, 100), # Bico de Pato / Mergulho
        (-50, 150), (0, 100), (-50, -50) # Junção / Subida (Fechamento será suavizado)
    ]
    
    # 2. BRASÍLIA - NELSON PIQUET (Anel externo puro + Miolo, Perímetro ~5489m)
    brasilia_pts = [
        (0, 0), (300, 0), (800, 0), (1100, 50), (1200, 200), (1100, 500), # Reta box e curva 1/2
        (800, 700), (500, 650), (200, 600), (0, 700), (-200, 650), # Miolo
        (-400, 500), (-600, 300), (-800, 100), (-800, -200), (-600, -400), # Curva Norte
        (-300, -400), (-100, -300) # Volta pra reta
    ]
    
    # 3. VELOCITTA (Traçado técnico de Mogi Guaçu)
    velocitta_pts = [
        (0,0), (150, 10), (250, 100), (200, 200), (100, 150), (50, 250),
        (-50, 300), (-150, 200), (-100, 50), (-200, 0), (-250, -100), (-100, -150)
    ]
    
    tracks_to_build = {
        "interlagos_offline": {"name": "Interlagos (Real Interpolado)", "pts": interlagos_pts},
        "brasilia_offline": {"name": "Brasilia (Real Interpolado)", "pts": brasilia_pts},
        "velocitta_offline": {"name": "Velocitta (Real Interpolado)", "pts": velocitta_pts}
    }
    
    for key, data in tracks_to_build.items():
        file_path = tracks_dir / f"{key}.hdf5"
        circuit = create_track_from_waypoints(data["name"], data["pts"], num_points=2000)
        
        try:
            writer = CircuitHDF5Writer(str(file_path))
            writer.write_circuit(circuit)
            print(f"✓ Construída pista {data['name']} com {len(circuit.centerline_x)} nós.")
            print(f"  -> Salvo em: {file_path}")
        except Exception as e:
            print(f"Erro ao salvar {key}: {e}")

if __name__ == "__main__":
    print("--- GERADOR OFFLINE DE PISTAS REAIS ---")
    print("Mapeando traçados geométricos via interpolação de Splines...")
    build_offline_tracks()
    print("Concluído! Pronto para simulação.")
