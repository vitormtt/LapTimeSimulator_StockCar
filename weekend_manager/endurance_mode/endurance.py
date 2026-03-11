"""Gestão da estratégia para a corrida Endurance de 3 horas (Etapa 10 — Goiânia).

Especificidades do formato Endurance Stock Car:
- Duração: 3 horas
- Troca obrigatória de pilotos
- Múltiplos pit stops — estratégia de combustível e pneus crítica
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stint:
    """Representa um stint (período de condução) na corrida Endurance."""
    driver: str
    start_lap: int
    end_lap: Optional[int] = None
    start_time_s: float = 0.0
    end_time_s: Optional[float] = None
    tire_set_id: Optional[int] = None
    fuel_start_l: Optional[float] = None
    fuel_end_l: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.end_time_s is not None:
            return self.end_time_s - self.start_time_s
        return None

    @property
    def laps(self) -> Optional[int]:
        if self.end_lap is not None:
            return self.end_lap - self.start_lap
        return None


class EnduranceStrategy:
    """Planeja e monitora a estratégia da corrida Endurance de 3 horas.

    Parameters
    ----------
    drivers : list[str]
        Lista de pilotos disponíveis para a corrida.
    race_duration_s : float
        Duração total da corrida em segundos (default: 10800 = 3h).
    """

    def __init__(
        self,
        drivers: list[str],
        race_duration_s: float = 10800.0,
    ) -> None:
        self.drivers = drivers
        self.race_duration_s = race_duration_s
        self.stints: list[Stint] = []

    def add_stint(self, stint: Stint) -> None:
        """Adiciona um stint ao plano de corrida."""
        self.stints.append(stint)

    def total_time_s(self) -> float:
        """Retorna o tempo total coberto pelos stints registrados."""
        return sum(s.duration_s or 0.0 for s in self.stints)

    def remaining_time_s(self) -> float:
        """Retorna o tempo restante de corrida."""
        return max(0.0, self.race_duration_s - self.total_time_s())

    def summary(self) -> str:
        """Retorna um resumo textual da estratégia."""
        lines = ["=== Estratégia Endurance 3h — Goiânia ===\n"]
        for i, stint in enumerate(self.stints, 1):
            lines.append(
                f"Stint {i}: {stint.driver} | "
                f"Voltas {stint.start_lap}–{stint.end_lap} | "
                f"Duração: {(stint.duration_s or 0)/60:.1f} min"
            )
        lines.append(f"\nTempo restante: {self.remaining_time_s()/60:.1f} min")
        return "\n".join(lines)
