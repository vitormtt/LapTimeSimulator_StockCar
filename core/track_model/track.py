"""Representação de um circuito a partir de arquivo de configuração YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrackConfig:
    """Parâmetros de um circuito carregados de arquivo .yaml."""

    name: str
    nickname: str
    city: str
    state: str
    lap_length_m: float
    turns: int
    direction: str
    altitude_m: float
    grip_nominal: float
    endurance_mode: bool = False
    night_race: bool = False
    sectors: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackConfig":
        """Carrega parâmetros do circuito a partir de um arquivo .yaml.

        Parameters
        ----------
        path : str | Path
            Caminho para o arquivo .yaml de configuração do circuito.

        Returns
        -------
        TrackConfig
            Instância populada com os dados do arquivo.
        """
        with open(path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f)

        t = data["track"]
        s = data["surface"]

        return cls(
            name=t["name"],
            nickname=t["nickname"],
            city=t["city"],
            state=t["state"],
            lap_length_m=t["lap_length_m"],
            turns=t["turns"],
            direction=t["direction"],
            altitude_m=t["altitude_m"],
            grip_nominal=s["grip_nominal"],
            endurance_mode=data.get("endurance_mode", False),
            night_race=t.get("night_race", False),
            sectors=data.get("sectors", []),
            raw=data,
        )
