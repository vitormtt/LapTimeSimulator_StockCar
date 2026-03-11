"""Varredura paramétrica: avalia o impacto de variações de setup no tempo de volta."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def parameter_sweep(
    sim_func: Callable[[dict], float],
    base_params: dict,
    param_name: str,
    values: np.ndarray,
) -> pd.DataFrame:
    """Executa varredura de um parâmetro e retorna tempo de volta em função do valor.

    Parameters
    ----------
    sim_func : Callable[[dict], float]
        Função que recebe um dicionário de parâmetros e retorna o tempo de volta (s).
    base_params : dict
        Parâmetros base do veículo.
    param_name : str
        Nome do parâmetro a variar (chave do dicionário).
    values : np.ndarray
        Vetor de valores a testar.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas [param_name, 'lap_time_s'].
    """
    results = []
    for v in values:
        params = {**base_params, param_name: v}
        lap_time = sim_func(params)
        results.append({param_name: v, "lap_time_s": lap_time})
    return pd.DataFrame(results)
