"""
Script para Download e Extração de Trajetórias do OpenStreetMap (OSM)
Foca na conversão Lat/Lon brutos para XY (metros) suavizados.
"""
import requests
import numpy as np
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tracks.circuit import CircuitData
from src.tracks.hdf5 import CircuitHDF5Writer

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância em metros entre duas coordenadas de GPS"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def download_osm_track(track_name: str, bbox: tuple, retry=0):
    print(f"[{track_name}] Consultando Overpass API do OpenStreetMap...")
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Timeout estendido para 50s. Adicionado recursividade para costurar trechos grandes.
    s, w, n, e = bbox
    overpass_query = f"""
    [out:json][timeout:50];
    (
      way["highway"="raceway"]({s},{w},{n},{e});
      way["highway"="racing"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão na API: {e}")
        return None
        
    if response.status_code == 504 and retry < 2:
        print(f"[{track_name}] Timeout 504 no satélite principal. Tentando endpoint alternativo e aguardando 5s...")
        time.sleep(5)
        overpass_url = "https://overpass.kumi.systems/api/interpreter" # Endpoint alternativo
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=60)
        
    if response.status_code != 200:
        print(f"[{track_name}] Erro na API: {response.status_code}")
        return None
        
    data = response.json()
    elements = data.get('elements', [])
    
    if not elements:
        print(f"[{track_name}] Nenhuma pista ('raceway') encontrada nesta região geográfica.")
        return None
        
    nodes = {el['id']: (el['lat'], el['lon']) for el in elements if el['type'] == 'node'}
    
    ways = [el for el in elements if el['type'] == 'way']
    if not ways:
        return None
        
    # Organiza para garantir que pegamos os trechos principais conectados e não apenas os pits
    ways = sorted(ways, key=lambda w: len(w['nodes']), reverse=True)
    main_way = ways[0]
    
    # Algoritmo de junção (Costurar loops como Interlagos e Brasília que as vezes vêm fatiados no OSM)
    all_track_nodes = []
    for node_id in main_way['nodes']:
        if node_id in nodes:
            all_track_nodes.append(nodes[node_id])
            
    # Tenta juntar o segundo maior segmento caso o primeiro não feche o autódromo
    if len(ways) > 1 and len(all_track_nodes) < 200:
        for node_id in ways[1]['nodes']:
            if node_id in nodes and nodes[node_id] not in all_track_nodes:
                all_track_nodes.append(nodes[node_id])
    
    lat0, lon0 = all_track_nodes[0]
    x_coords = [0.0]
    y_coords = [0.0]
    
    for i in range(1, len(all_track_nodes)):
        lat, lon = all_track_nodes[i]
        x_dist = haversine_distance(lat0, lon0, lat0, lon) * (1 if lon > lon0 else -1)
        y_dist = haversine_distance(lat0, lon0, lat, lon0) * (1 if lat > lat0 else -1)
        x_coords.append(x_dist)
        y_coords.append(y_dist)
        
    x_np = np.array(x_coords)
    y_np = np.array(y_coords)
    
    ds = np.sqrt(np.diff(x_np)**2 + np.diff(y_np)**2)
    total_length = np.sum(ds)
    
    print(f"[{track_name}] Download concluído! {len(x_np)} nós processados. Comprimento real lido: {total_length:.2f}m")
    
    track_width = np.full_like(x_np, 12.0)
    circuit = CircuitData(
        name=track_name,
        centerline_x=x_np,
        centerline_y=y_np,
        left_boundary_x=x_np,
        left_boundary_y=y_np + track_width/2,
        right_boundary_x=x_np,
        right_boundary_y=y_np - track_width/2,
        track_width=track_width
    )
    
    return circuit

def build_real_tracks():
    tracks_dir = ROOT_DIR / "tracks"
    os.makedirs(tracks_dir, exist_ok=True)
    
    # Bounding Boxes expandidas e adaptadas para garantir a malha correta do autódromo
    autodromos = {
        "interlagos_real": {"name": "Autódromo José Carlos Pace (Interlagos)", "bbox": (-23.706, -46.702, -23.696, -46.690)},
        "velocitta_real": {"name": "Velocitta", "bbox": (-22.287, -46.885, -22.274, -46.872)},
        "brasilia_real": {"name": "Autódromo Nelson Piquet (Brasília)", "bbox": (-15.776, -47.904, -15.762, -47.886)}
    }
    
    for key, data in autodromos.items():
        file_path = tracks_dir / f"{key}.hdf5"
        circuit = download_osm_track(data["name"], data["bbox"])
        
        if circuit:
            try:
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
