"""Cálculo do Throttle Application Index (TAI) por saída de curva.

O TAI avalia quão cedo e quão progressivamente o piloto aplica o acelerador
após o vértice da curva, impactando diretamente o tempo de saída.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tai(
    df: pd.DataFrame,
    throttle_col: str = "throttle_pct",
    distance_col: str = "distance_m",
) -> pd.DataFrame:
    """Calcula o gradiente de aplicação do acelerador ao longo da distância.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de telemetria.
    throttle_col : str
        Nome da coluna de posição do acelerador (%).
    distance_col : str
        Nome da coluna de distância acumulada (m).

    Returns
    -------
    pd.DataFrame
        DataFrame com coluna 'tai' (d_throttle/d_distance) adicionada.
    """
    dd = df[distance_col].diff().replace(0, np.nan)
    df["tai"] = df[throttle_col].diff() / dd
    return df
