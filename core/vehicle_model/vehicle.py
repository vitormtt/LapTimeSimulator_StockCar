"""Representação do veículo a partir de arquivo de configuração YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VehicleConfig:
    """Parâmetros completos de um veículo SNG01 carregados de arquivo .yaml."""

    name: str
    manufacturer: str
    generation: str
    season: int
    mass_total_kg: float
    wheelbase_m: float
    cg_height_m: float
    cg_bias_front: float
    max_power_kw: float
    max_torque_nm: float
    gear_ratios: list[float]
    final_drive_ratio: float
    cx: float
    frontal_area_m2: float
    cl_front: float
    cl_rear: float
    tire_radius_m: float
    brake_bias_front: float
    max_decel_g: float
    spring_rate_front_nm: float
    spring_rate_rear_nm: float
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleConfig":
        """Carrega parâmetros do veículo a partir de um arquivo .yaml.

        Parameters
        ----------
        path : str | Path
            Caminho para o arquivo .yaml de configuração do veículo.

        Returns
        -------
        VehicleConfig
            Instância populada com os parâmetros do arquivo.
        """
        with open(path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f)

        v = data["vehicle"]
        m = data["mass"]
        g = data["geometry"]
        p = data["powertrain"]
        a = data["aerodynamics"]
        t = data["tires"]
        s = data["suspension"]
        b = data["brakes"]

        return cls(
            name=v["name"],
            manufacturer=v["manufacturer"],
            generation=v["generation"],
            season=v["season"],
            mass_total_kg=m["total_kg"],
            wheelbase_m=g["wheelbase_m"],
            cg_height_m=g["cg_height_m"],
            cg_bias_front=g["cg_bias_front"],
            max_power_kw=p["max_power_kw"],
            max_torque_nm=p["max_torque_nm"],
            gear_ratios=p["gear_ratios"],
            final_drive_ratio=p["final_drive_ratio"],
            cx=a["cx"],
            frontal_area_m2=a["frontal_area_m2"],
            cl_front=a["cl_front"],
            cl_rear=a["cl_rear"],
            tire_radius_m=t["radius_m"],
            brake_bias_front=b["bias_front"],
            max_decel_g=b["max_decel_g"],
            spring_rate_front_nm=s["spring_rate_front_nm"],
            spring_rate_rear_nm=s["spring_rate_rear_nm"],
            raw=data,
        )
