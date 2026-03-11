"""Geração do relatório pós-etapa em formato PDF e CSV.

Conteúdo padrão do relatório pós-etapa:
  1. KPIs consolidados de todas as sessões (TL1, TL2, Q, Sprint, Race)
  2. Comparativo tempo simulado vs. tempo real
  3. Análise de pilotagem (BEI, TAI, gg-diagram)
  4. Evolução de setup entre sessões
  5. Resumo de estratégia de pit (tempo de pit, pneus utilizados)
  6. Pontuação e posições finais
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class PostEventReport:
    """Gera o relatório pós-etapa a partir dos dados de todas as sessões.

    Parameters
    ----------
    event_round : int
        Número da etapa (1–12).
    track_nickname : str
        Apelido do circuito.
    sessions_data : dict[str, pd.DataFrame]
        Dicionário com DataFrames de telemetria por sessão.
        Chaves esperadas: 'TL1', 'TL2', 'Q1', 'Q2', 'SPRINT', 'RACE'.
    output_dir : Path
        Diretório de saída.
    """

    def __init__(
        self,
        event_round: int,
        track_nickname: str,
        sessions_data: dict[str, pd.DataFrame],
        output_dir: Path = Path("outputs/reports/post_event"),
    ) -> None:
        self.event_round = event_round
        self.track_nickname = track_nickname
        self.sessions_data = sessions_data
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_csv(self) -> Path:
        """Exporta KPIs consolidados em CSV."""
        # TODO: Implementar com kpi_engine.compute_all_kpis por sessão
        raise NotImplementedError

    def generate_pdf(self) -> Path:
        """Gera relatório PDF via ReportLab com layout de etapa."""
        # TODO: Implementar layout completo do relatório pós-etapa
        raise NotImplementedError
