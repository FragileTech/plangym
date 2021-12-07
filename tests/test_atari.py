import pytest

from plangym.atari import AtariEnvironment
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def pacman_obs():
    return AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)


def qbert_ram():
    return AtariEnvironment(name="Qbert-ram-v0", clone_seeds=False, autoreset=False)


def pong_obs_ram():
    return AtariEnvironment(name="PongDeterministic-v4")


def qbert_new_ale():
    return AtariEnvironment(name="ALE/Qbert-v5")


environments = [pacman_obs, qbert_ram, pong_obs_ram, qbert_new_ale]


@pytest.fixture(params=environments, scope="class")
def env(request) -> AtariEnvironment:
    return request.param()
