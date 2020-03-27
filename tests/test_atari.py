import pytest

from plangym.atari import AtariEnvironment
from tests.test_core import TestGymEnvironment


def pacman_obs():
    return AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)


def qbert_ram():
    return AtariEnvironment(name="Qbert-ram-v0", clone_seeds=False, autoreset=False)


def pong_obs_ram():
    return AtariEnvironment(name="PongDeterministic-v4", obs_ram=True)


environments = [pacman_obs, qbert_ram, pong_obs_ram]


@pytest.fixture(params=environments, scope="class")
def env(request) -> AtariEnvironment:
    return request.param()
