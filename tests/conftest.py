from __future__ import annotations

import pytest

from app import create_app


@pytest.fixture()
def app():
    return create_app({"TESTING": True, "AUTO_REFRESH": False})


@pytest.fixture()
def client(app):
    return app.test_client()

