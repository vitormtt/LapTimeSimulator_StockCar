"""Overlay de voltas: alinhamento por distância e cálculo de delta de tempo."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def align_laps_by_distance(
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
    distance_col: str = "distance_m",
    n_points: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpola duas voltas sobre uma grade de distância comum.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Volta de referência.
    df_comp : pd.DataFrame
        Volta de comparação.
    distance_col : str
        Coluna de distância acumulada.
    n_points : int
        Número de pontos da grade comum de distância.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Par de DataFrames interpolados sobre a mesma grade.
    """
    # TODO: Implementar interpolação e alinhamento
    raise NotImplementedError("Implementar no Módulo de Análise de Pilotagem.")
