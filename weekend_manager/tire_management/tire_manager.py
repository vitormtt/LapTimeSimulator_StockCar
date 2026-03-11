"""Gestão de sets de pneus conforme regulamento BRB Stock Car Pro Series 2026.

Regulamento relevante:
- Fornecedor único: Hankook
- Pit stop obrigatório na Corrida Principal: troca mínima de 2 pneus
- Sets disponíveis por final de semana: a validar com regulamento técnico 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TireCompound(Enum):
    """Compostos de pneu disponíveis na Stock Car."""
    SLICK = "slick"
    WET = "wet"
    INTERMEDIATE = "intermediate"  # Se aplicável ao regulamento


@dataclass
class TireSet:
    """Representa um set de pneus (4 pneus)."""
    set_id: int
    compound: TireCompound
    laps_used: int = 0
    is_scrubbed: bool = False   # Pneu rodado (1 volta de aquecimento)
    session_used: Optional[str] = None

    @property
    def is_new(self) -> bool:
        return self.laps_used == 0


class TireManager:
    """Controla o estoque e uso de pneus ao longo do final de semana.

    Parameters
    ----------
    total_sets : int
        Total de sets disponíveis para o final de semana.
    """

    def __init__(self, total_sets: int = 8) -> None:
        self.sets: list[TireSet] = [
            TireSet(set_id=i + 1, compound=TireCompound.SLICK)
            for i in range(total_sets)
        ]

    def available_sets(self) -> list[TireSet]:
        """Retorna sets ainda não utilizados (novos)."""
        return [s for s in self.sets if s.is_new]

    def use_set(self, set_id: int, laps: int, session: str) -> None:
        """Registra uso de um set de pneus em uma sessão."""
        for s in self.sets:
            if s.set_id == set_id:
                s.laps_used += laps
                s.session_used = session
                break

    def pit_strategy(
        self,
        current_set_id: int,
        tires_to_change: int = 2,
    ) -> str:
        """Sugere estratégia de pit stop conforme regulamento.

        Parameters
        ----------
        current_set_id : int
            Set de pneus atual no carro.
        tires_to_change : int
            Número de pneus a trocar (mínimo 2 por regulamento).

        Returns
        -------
        str
            Descrição da estratégia de pit.
        """
        available = self.available_sets()
        if not available:
            return "ALERTA: Nenhum set novo disponível para pit stop!"
        best = available[0]
        return (
            f"Pit strategy: trocar {tires_to_change} pneus. "
            f"Set atual: #{current_set_id} → Novo set sugerido: #{best.set_id} "
            f"({best.compound.value})"
        )
