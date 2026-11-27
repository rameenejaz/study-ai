import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "sample_data"


def _read(name: str) -> str:
    return (SAMPLES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def os_notes() -> str:
    return _read("os_lecture_2000_words.txt")


@pytest.fixture(scope="session")
def ml_notes() -> str:
    return _read("machine_learning_notes.txt")


@pytest.fixture(scope="session")
def business_notes() -> str:
    return _read("business_management_notes.txt")


@pytest.fixture(scope="session")
def football_notes() -> str:
    return _read("football_sports_notes.txt")


@pytest.fixture(scope="session")
def short_notes() -> str:
    return _read("short_note_118_words.txt")


@pytest.fixture(scope="session")
def all_domain_notes(os_notes, ml_notes, business_notes, football_notes):
    return {"os": os_notes, "ml": ml_notes, "business": business_notes, "football": football_notes}
