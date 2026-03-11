"""Comparação de desempenho entre sessões (TL1, TL2, Q, Sprint, Race)."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def best_lap(df: pd.DataFrame, time_col: str = "time_s", lap_col: str = "lap_number") -> pd.DataFrame:
    """Retorna os dados da volta mais rápida de um DataFrame de telemetria.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo de uma sessão.
    time_col : str
        Nome da coluna de tempo.
    lap_col : str
        Nome da coluna de número de volta.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado para a volta mais rápida.
    """
    lap_times = df.groupby(lap_col)[time_col].max() - df.groupby(lap_col)[time_col].min()
    fastest_lap = lap_times.idxmin()
    return df[df[lap_col] == fastest_lap].reset_index(drop=True)


def lap_delta(
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
    distance_col: str = "distance_m",
    time_col: str = "time_s",
) -> pd.DataFrame:
    """Calcula o delta de tempo entre duas voltas ao longo da distância.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Volta de referência (ex: melhor volta do qualy).
    df_comp : pd.DataFrame
        Volta de comparação.
    distance_col : str
        Coluna de distância acumulada.
    time_col : str
        Coluna de tempo.

    Returns
    -------
    pd.DataFrame
        DataFrame com coluna 'delta_s' ao longo da distância.
    """
    # TODO: Implementar interpolação sobre distância comum
    raise NotImplementedError("Implementar no Módulo de Análise de Pilotagem.")
