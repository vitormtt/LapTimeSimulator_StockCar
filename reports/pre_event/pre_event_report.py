"""Geração do relatório pré-etapa em formato PDF e CSV.

Conteúdo padrão do relatório pré-etapa:
  1. Dados da pista (circuito, comprimento, setores, altitude)
  2. Configuração base do veículo para o circuito
  3. Tempo de volta simulado de referência
  4. Estratégia de pneus sugerida
  5. Histórico de desempenho na pista (etapas anteriores)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.track_model.track import TrackConfig
from core.vehicle_model.vehicle import VehicleConfig


class PreEventReport:
    """Gera o relatório pré-etapa para um evento da Stock Car.

    Parameters
    ----------
    track : TrackConfig
        Configuração do circuito da etapa.
    vehicle : VehicleConfig
        Configuração do veículo.
    output_dir : Path
        Diretório de saída para os arquivos gerados.
    """

    def __init__(
        self,
        track: TrackConfig,
        vehicle: VehicleConfig,
        output_dir: Path = Path("outputs/reports/pre_event"),
    ) -> None:
        self.track = track
        self.vehicle = vehicle
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_csv(self) -> Path:
        """Exporta dados do relatório em formato CSV."""
        # TODO: Implementar geração de CSV com pandas
        raise NotImplementedError

    def generate_pdf(self) -> Path:
        """Gera relatório PDF via ReportLab."""
        # TODO: Implementar layout do relatório com ReportLab
        raise NotImplementedError
