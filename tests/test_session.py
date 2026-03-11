"""Testes unitários para o weekend_manager (SessionRecord e SessionType)."""

from weekend_manager.session_logger.session import SessionRecord, SessionType


def test_session_record_creation():
    s = SessionRecord(
        session_type=SessionType.FREE_PRACTICE_1,
        event_round=3,
        track_nickname="Interlagos",
        driver="Piloto Teste",
        vehicle="Chevrolet Tracker",
        car_number=10,
    )
    assert s.session_type == SessionType.FREE_PRACTICE_1
    assert s.event_round == 3


def test_session_types_count():
    """Verifica que todos os 8 tipos de sessão estão definidos."""
    assert len(SessionType) == 8
