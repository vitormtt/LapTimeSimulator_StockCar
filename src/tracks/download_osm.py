"""
Script para Download e Extração de Trajetórias do OpenStreetMap (OSM)
Foca na conversão Lat/Lon brutos para XY (metros) suavizados.
"""
import requests
import numpy as np
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tracks.circuit import Circuit
from src.tracks.hdf5 import CircuitHDF5Writer

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância em metros entre duas coordenadas de GPS"""
    R = 6371000  # Raio da Terra em metros
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def download_osm_track(track_name: str, bbox: tuple):
    """
    Usa a API pública do Overpass para buscar malhas de autódromos usando Bounding Box (S, W, N, E).
    Isso substitui a necessidade do pacote complexo osmnx no Windows.
    """
    print(f"[{track_name}] Consultando Overpass API do OpenStreetMap...")
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query para buscar vias do tipo "racing" ou "track" no bbox informado
    s, w, n, e = bbox
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["highway"="racing"]({s},{w},{n},{e});
      way["highway"="track"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    if response.status_code != 200:
        print(f"Erro na API: {response.status_code}")
        return None
        
    data = response.json()
    elements = data.get('elements', [])
    
    if not elements:
        print(f"[{track_name}] Nenhuma pista encontrada nesta região geográfica.")
        return None
        
    # Processa os Nodes
    nodes = {el['id']: (el['lat'], el['lon']) for el in elements if el['type'] == 'node'}
    
    # Tenta achar o Way mais longo (geralmente a pista principal)
    ways = [el for el in elements if el['type'] == 'way']
    if not ways:
        return None
        
    ways = sorted(ways, key=lambda w: len(w['nodes']), reverse=True)
    main_way = ways[0]
    
    track_nodes = [nodes[node_id] for node_id in main_way['nodes'] if node_id in nodes]
    
    # Converte Lat/Lon para X/Y (Metros) locais, assumindo o primeiro ponto como origem (0,0)
    lat0, lon0 = track_nodes[0]
    x_coords = [0.0]
    y_coords = [0.0]
    
    for i in range(1, len(track_nodes)):
        lat, lon = track_nodes[i]
        # X é Distância na Longitude, Y é distância na Latitude
        x_dist = haversine_distance(lat0, lon0, lat0, lon) * (1 if lon > lon0 else -1)
        y_dist = haversine_distance(lat0, lon0, lat, lon0) * (1 if lat > lat0 else -1)
        x_coords.append(x_dist)
        y_coords.append(y_dist)
        
    # Calcula comprimento
    x_np = np.array(x_coords)
    y_np = np.array(y_coords)
    
    ds = np.sqrt(np.diff(x_np)**2 + np.diff(y_np)**2)
    total_length = np.sum(ds)
    
    print(f"[{track_name}] Download concluído! {len(x_np)} nós processados. Comprimento: {total_length:.2f}m")
    
    # Cria o objeto Circuit
    track_width = 12.0
    circuit = Circuit(
        name=track_name,
        length=total_length,
        centerline_x=x_np,
        centerline_y=y_np,
        left_boundary_x=x_np,
        left_boundary_y=y_np + track_width/2,
        right_boundary_x=x_np,
        right_boundary_y=y_np - track_width/2
    )
    
    return circuit

def build_real_tracks():
    tracks_dir = ROOT_DIR / "tracks"
    os.makedirs(tracks_dir, exist_ok=True)
    
    # Bounding Boxes aproximados para Interlagos e Velocitta (South, West, North, East)
    # Para capturar todo o traçado
    autodromos = {
        "interlagos_real": {"name": "Autódromo José Carlos Pace (Interlagos)", "bbox": (-23.708, -46.703, -23.695, -46.688)},
        "velocitta_real": {"name": "Velocitta", "bbox": (-22.285, -46.885, -22.270, -46.870)},
        "brasilia_real": {"name": "Autódromo Nelson Piquet (Brasília)", "bbox": (-15.772, -47.905, -15.755, -47.885)}
    }
    
    for key, data in autodromos.items():
        file_path = tracks_dir / f"{key}.hdf5"
        circuit = download_osm_track(data["name"], data["bbox"])
        
        if circuit:
            try:
                # Tenta gravar em HDF5 se h5py estiver instalado
                writer = CircuitHDF5Writer(str(file_path))
                writer.write_circuit(circuit)
                print(f"✓ Salvo HDF5 Real em: {file_path}\n")
            except Exception as e:
                print(f"Aviso: Não salvou {key} em HDF5 ({e}).")

if __name__ == "__main__":
    print("--- EXTRATOR DE PISTAS REAIS DO OPENSTREETMAP ---")
    print("Buscando vetores matemáticos e curvaturas geolocalizadas...")
    build_real_tracks()
    print("Concluído!")
