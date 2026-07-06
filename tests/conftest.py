"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deploy.mock_engine.main import app as mock_app


@pytest.fixture(scope="session")
def mock_engine_client() -> TestClient:
    """Synchronous test client for the mock inference engine."""
    return TestClient(mock_app)
