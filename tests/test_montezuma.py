import pytest

from plangym.montezuma import Montezuma
from tests.test_core import TestGymEnvironment


@pytest.fixture(scope="class")
def env() -> Montezuma:
    return Montezuma(autoreset=True)
