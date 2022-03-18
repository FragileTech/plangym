import pytest

from plangym.minimal import AtariEnvironment, MinimalPacman, MinimalPong
from tests import SKIP_ATARI_TESTS


if SKIP_ATARI_TESTS:
    pytest.skip("Atari not installed, skipping", allow_module_level=True)
from plangym.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def minimal_pacman():
    return MinimalPacman(clone_seeds=True, autoreset=True)


def minimal_pong():
    return MinimalPong(clone_seeds=False, autoreset=False)


def minimal_pong_ram():
    return MinimalPong(name="Pong-ram-v0")


environments = [minimal_pacman, minimal_pong, minimal_pong_ram]


@pytest.fixture(params=environments, scope="class")
def env(request) -> AtariEnvironment:
    return request.param()
