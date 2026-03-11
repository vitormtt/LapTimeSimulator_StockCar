"""Testes unitários para o módulo de modelo de veículo."""

from pathlib import Path

import pytest

from core.vehicle_model.vehicle import VehicleConfig

CONFIG_DIR = Path("config/vehicles")


def test_load_tracker():
    """Verifica carregamento correto do arquivo tracker.yaml."""
    v = VehicleConfig.from_yaml(CONFIG_DIR / "tracker.yaml")
    assert v.name == "Chevrolet Tracker"
    assert v.mass_total_kg == 1100.0
    assert v.max_power_kw > 0


def test_load_corolla_cross():
    v = VehicleConfig.from_yaml(CONFIG_DIR / "corolla_cross.yaml")
    assert v.manufacturer == "Toyota"


def test_load_eclipse_cross():
    v = VehicleConfig.from_yaml(CONFIG_DIR / "eclipse_cross.yaml")
    assert v.manufacturer == "Mitsubishi"
