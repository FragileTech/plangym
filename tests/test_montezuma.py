import pytest

from plangym.montezuma import Montezuma
from tests.api_tests import batch_size, TestGymEnvironment


@pytest.fixture(scope="class")
def env() -> Montezuma:
    return Montezuma(autoreset=True)
