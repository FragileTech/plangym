import pytest

from plangym.montezuma import Montezuma
from tests.test_core import TestGymEnvironment


@pytest.fixture()
def env() -> Montezuma:
    return Montezuma(autoreset=True)
