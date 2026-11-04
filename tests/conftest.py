"""Shared fixtures for the reliability lab tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from arl.db.catalog import FIXED_TODAY
from arl.db.repository import OpsRepository, connect_sqlite
from arl.db.seed import seed


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[OpsRepository]:
    database = OpsRepository(connect_sqlite(str(tmp_path / "ops.db"), FIXED_TODAY))
    seed(database)
    yield database
    database.db.close()
