"""Cálculo do Braking Efficiency Index (BEI) por zona de frenagem.

O BEI mede a eficiência com que o piloto utiliza a capacidade máxima de
desaceleração do veículo em cada frenagem ao longo da volta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bei(
    df: pd.DataFrame,
    ax_col: str = "ax_g",
    max_decel_g: float = 2.2,
    braking_flag_col: str = "is_braking_zone",
) -> pd.DataFrame:
    """Calcula o BEI para cada ponto de frenagem identificado.

    BEI = |ax_real| / ax_max  (adimensional, 0–1)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de telemetria com canal de aceleração longitudinal.
    ax_col : str
        Nome da coluna de aceleração longitudinal (g).
    max_decel_g : float
        Desaceleração máxima do veículo (g) — parâmetro do veículo.
    braking_flag_col : str
        Coluna booleana indicando zona de frenagem ativa.

    Returns
    -------
    pd.DataFrame
        DataFrame com coluna 'bei' adicionada.
    """
    df["bei"] = np.where(
        df[braking_flag_col],
        df[ax_col].abs() / max_decel_g,
        np.nan,
    )
    return df
