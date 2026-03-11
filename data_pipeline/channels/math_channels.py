"""Canais matemáticos derivados dos dados brutos de telemetria.

Todos os métodos recebem um DataFrame normalizado (saída do MoTecImporter)
e retornam o mesmo DataFrame com as novas colunas adicionadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_velocity_ms(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona canal de velocidade em m/s a partir de km/h."""
    df["velocity_ms"] = df["velocity_kmh"] / 3.6
    return df


def add_gg_resultant(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona aceleração resultante (gg-diagram magnitude) em g."""
    df["g_resultant"] = np.sqrt(df["ax_g"] ** 2 + df["ay_g"] ** 2)
    return df


def add_distance_from_time(
    df: pd.DataFrame,
    time_col: str = "time_s",
    velocity_col: str = "velocity_ms",
) -> pd.DataFrame:
    """Calcula distância acumulada por integração trapezoidal de velocidade."""
    dt = df[time_col].diff().fillna(0.0)
    df["distance_integrated_m"] = (df[velocity_col] * dt).cumsum()
    return df


def add_braking_zones(
    df: pd.DataFrame,
    threshold_ax_g: float = -0.5,
) -> pd.DataFrame:
    """Marca pontos de frenagem intensa (ax < threshold) com flag booleana.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de telemetria.
    threshold_ax_g : float
        Limiar de desaceleração em g para identificar zona de frenagem.
    """
    df["is_braking_zone"] = df["ax_g"] < threshold_ax_g
    return df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos os canais matemáticos ao DataFrame."""
    df = add_velocity_ms(df)
    df = add_gg_resultant(df)
    df = add_distance_from_time(df)
    df = add_braking_zones(df)
    return df
