"""Estrutura de dados para sessões do final de semana BRB Stock Car Pro Series.

Formato padrão do final de semana (regulamento CBA 2026):
  Sexta-feira : Treino Livre 1 (TL1), Treino Livre 2 (TL2)
  Sábado      : Top Qualifying (Q1 + Q2), Sprint Race (~30 min + 1 volta)
  Domingo     : Warm-up, Corrida Principal (~50 min + 1 volta, 1 pit obrigatório)

Exceção: Etapa Goiânia Endurance (Etapa 10) — corrida de 3h com troca de pilotos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class SessionType(Enum):
    """Tipos de sessão do final de semana Stock Car."""
    FREE_PRACTICE_1 = "TL1"
    FREE_PRACTICE_2 = "TL2"
    QUALIFYING_1 = "Q1"
    QUALIFYING_2 = "Q2"
    SPRINT_RACE = "SPRINT"
    WARMUP = "WARMUP"
    MAIN_RACE = "RACE"
    ENDURANCE = "ENDURANCE"   # Exclusivo etapa Goiânia


@dataclass
class SessionRecord:
    """Registro de uma sessão do final de semana."""

    session_type: SessionType
    event_round: int                     # Número da etapa (1–12)
    track_nickname: str                  # Ex: 'Interlagos', 'Velocitta'
    driver: str                          # Nome do piloto
    vehicle: str                         # Ex: 'Chevrolet Tracker'
    car_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    best_lap_s: Optional[float] = None
    laps_completed: int = 0
    telemetry_path: Optional[Path] = None
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"[Etapa {self.event_round:02d} | {self.track_nickname} | "
            f"{self.session_type.value}] {self.driver} — #{self.car_number} "
            f"{self.vehicle} | Best: {self.best_lap_s:.3f}s" if self.best_lap_s
            else f"[Etapa {self.event_round:02d} | {self.track_nickname} | "
            f"{self.session_type.value}] {self.driver} — #{self.car_number} {self.vehicle}"
        )
