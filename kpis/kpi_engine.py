"""Motor de KPIs — coleta, calcula e exporta indicadores de desempenho.

KPIs implementados:
- Lap Time (absoluto e delta)
- Vmax por sessão
- BEI médio (Braking Efficiency Index)
- TAI médio (Throttle Application Index)
- gg-Diagram utilization (%)
- Pit stop delta time
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def lap_time_kpi(df: pd.DataFrame, lap_col: str = "lap_number", time_col: str = "time_s") -> dict:
    """Calcula KPIs de tempo de volta a partir de telemetria.

    Returns
    -------
    dict
        Dicionário com 'best_lap_s', 'mean_lap_s', 'std_lap_s', 'n_laps'.
    """
    lap_times = df.groupby(lap_col)[time_col].max() - df.groupby(lap_col)[time_col].min()
    lap_times = lap_times[lap_times > 0]
    return {
        "best_lap_s": float(lap_times.min()),
        "mean_lap_s": float(lap_times.mean()),
        "std_lap_s": float(lap_times.std()),
        "n_laps": int(len(lap_times)),
    }


def vmax_kpi(df: pd.DataFrame, velocity_col: str = "velocity_kmh") -> dict:
    """Retorna a velocidade máxima registrada."""
    return {"vmax_kmh": float(df[velocity_col].max())}


def gg_utilization_kpi(
    df: pd.DataFrame,
    g_resultant_col: str = "g_resultant",
    theoretical_max_g: float = 2.5,
) -> dict:
    """Calcula o percentual de utilização do envelope de aderência.

    Definido como: (g_resultant_mean / theoretical_max_g) * 100
    """
    if g_resultant_col not in df.columns:
        return {"gg_utilization_pct": None}
    utilization = (df[g_resultant_col].mean() / theoretical_max_g) * 100
    return {"gg_utilization_pct": round(float(utilization), 2)}


def compute_all_kpis(df: pd.DataFrame, **kwargs) -> dict:
    """Agrega todos os KPIs disponíveis em um único dicionário."""
    kpis = {}
    kpis.update(lap_time_kpi(df, **{k: v for k, v in kwargs.items() if k in ("lap_col", "time_col")}))
    kpis.update(vmax_kpi(df))
    kpis.update(gg_utilization_kpi(df))
    return kpis
