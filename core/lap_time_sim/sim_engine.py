"""Engine principal de simulação quasi-estática de tempo de volta.

Referências
-----------
- Hakewill, J. (2010). Lap Time Simulation Model for Racing Cars.
- Pagano, S. et al. (2016). Lap Time Simulation for Performance Vehicle Development.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.vehicle_model.vehicle import VehicleConfig
from core.track_model.track import TrackConfig


class LapTimeSimulator:
    """Simulador quasi-estático de tempo de volta.

    Parameters
    ----------
    vehicle : VehicleConfig
        Configuração do veículo.
    track : TrackConfig
        Configuração do circuito.
    n_points : int, optional
        Número de pontos de discretização da pista (default: 1000).
    """

    RHO_AIR: float = 1.225  # Densidade do ar ao nível do mar (kg/m³)
    G: float = 9.81         # Aceleração gravitacional (m/s²)

    def __init__(
        self,
        vehicle: VehicleConfig,
        track: TrackConfig,
        n_points: int = 1000,
    ) -> None:
        self.vehicle = vehicle
        self.track = track
        self.n_points = n_points
        self._results: Optional[dict] = None

    def run(self) -> dict:
        """Executa a simulação e retorna os resultados.

        Returns
        -------
        dict
            Dicionário com canais: 'distance', 'velocity', 'lap_time', etc.
        """
        # TODO: Implementar perfil de velocidade quasi-estático
        # Etapas: 1) Calcular Vmax por ponto (limitado por aderência, potência, aerodinâmica)
        #         2) Integrar perfil de velocidade com restrições de aceleração e frenagem
        #         3) Integrar tempo de volta
        raise NotImplementedError("Engine de simulação será implementada no Módulo 4.")

    @property
    def results(self) -> Optional[dict]:
        """Retorna os resultados da última simulação."""
        return self._results

    def _calc_aero_forces(self, velocity_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calcula arrasto e downforce em função da velocidade.

        Parameters
        ----------
        velocity_ms : np.ndarray
            Vetor de velocidades (m/s).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (drag_N, downforce_N)
        """
        q = 0.5 * self.RHO_AIR * velocity_ms**2  # Pressão dinâmica (Pa)
        drag = q * self.vehicle.cx * self.vehicle.frontal_area_m2
        cl_total = self.vehicle.cl_front + self.vehicle.cl_rear
        downforce = q * cl_total * self.vehicle.frontal_area_m2
        return drag, downforce
