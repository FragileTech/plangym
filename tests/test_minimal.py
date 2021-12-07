import pytest

from plangym.minimal import AtariEnvironment, MinimalPacman, MinimalPong
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def minimal_pacman():
    return MinimalPacman(clone_seeds=True, autoreset=True)


def minimal_pong():
    return MinimalPong(clone_seeds=False, autoreset=False)


environments = [minimal_pacman, minimal_pong]


@pytest.fixture(params=environments, scope="class")
def env(request) -> AtariEnvironment:
    return request.param()
