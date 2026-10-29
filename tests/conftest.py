"""Keep environment-based runtime options isolated between offline tests."""
import os

import pytest


@pytest.fixture(autouse=True)
def restore_runtime_environment():
    previous = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(previous)
