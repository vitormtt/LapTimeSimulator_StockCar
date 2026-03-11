"""Testes unitários para o módulo de modelo de pista."""

from pathlib import Path

import pytest

from core.track_model.track import TrackConfig

CONFIG_DIR = Path("config/tracks")


def test_load_interlagos():
    t = TrackConfig.from_yaml(CONFIG_DIR / "interlagos.yaml")
    assert t.nickname == "Interlagos"
    assert t.lap_length_m > 0
    assert t.endurance_mode is False


def test_load_goiania_endurance():
    t = TrackConfig.from_yaml(CONFIG_DIR / "goiania.yaml")
    assert t.endurance_mode is True


def test_load_chapeco_grip_penalty():
    """Chapecó é pista nova — grip_nominal deve ser < 1.0."""
    t = TrackConfig.from_yaml(CONFIG_DIR / "chapeco.yaml")
    assert t.grip_nominal < 1.0
