"""Importador de telemetria MoTec (.csv) para DataFrame pandas.

Formato esperado: exportação padrão do MoTec i2 Pro (cabeçalho com metadados
nas primeiras linhas, seguido de dados tabulares com canais como colunas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# Mapeamento de nomes de canais MoTec → nomes internos padronizados
CHANNEL_MAP: dict[str, str] = {
    "Speed": "velocity_kmh",
    "RPM": "engine_rpm",
    "Gear": "gear",
    "ThrottlePos": "throttle_pct",
    "BrakePres": "brake_pressure_bar",
    "SteeringAngle": "steering_deg",
    "LateralAccel": "ay_g",
    "LongAccel": "ax_g",
    "Lap": "lap_number",
    "Time": "time_s",
    "Distance": "distance_m",
}


class MoTecImporter:
    """Importa e normaliza dados de telemetria exportados pelo MoTec i2 Pro.

    Parameters
    ----------
    path : str | Path
        Caminho para o arquivo .csv de telemetria.
    header_rows : int, optional
        Número de linhas de cabeçalho a ignorar (default: 13).
    channel_map : dict[str, str], optional
        Mapeamento customizado de nomes de canais. Usa CHANNEL_MAP por padrão.
    """

    def __init__(
        self,
        path: str | Path,
        header_rows: int = 13,
        channel_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.path = Path(path)
        self.header_rows = header_rows
        self.channel_map = channel_map or CHANNEL_MAP
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Carrega e normaliza o arquivo .csv de telemetria.

        Returns
        -------
        pd.DataFrame
            DataFrame com canais normalizados e tipos corretos.

        Raises
        ------
        FileNotFoundError
            Se o arquivo não for encontrado no caminho especificado.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Arquivo de telemetria não encontrado: {self.path}")

        self._df = pd.read_csv(
            self.path,
            skiprows=self.header_rows,
            low_memory=False,
        )

        # Renomeia colunas conforme mapeamento
        self._df.rename(columns=self.channel_map, inplace=True)

        # Converte para tipos numéricos onde aplicável
        for col in self._df.columns:
            self._df[col] = pd.to_numeric(self._df[col], errors="ignore")

        return self._df

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Retorna o DataFrame carregado (None se load() não foi chamado)."""
        return self._df

    def get_lap(self, lap_number: int) -> pd.DataFrame:
        """Filtra os dados de uma volta específica.

        Parameters
        ----------
        lap_number : int
            Número da volta a filtrar.

        Returns
        -------
        pd.DataFrame
            Subconjunto do DataFrame para a volta solicitada.
        """
        if self._df is None:
            raise RuntimeError("Chame load() antes de filtrar voltas.")
        return self._df[self._df["lap_number"] == lap_number].reset_index(drop=True)
