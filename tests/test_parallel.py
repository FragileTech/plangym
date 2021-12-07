import pytest

from plangym.atari import AtariEnvironment
from plangym.classic_control import ClassicControl
from plangym.parallel import ParallelEnvironment
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def parallel_cartpole():
    return ParallelEnvironment(env_class=ClassicControl, name="CartPole-v0")


def parallel_pacman():
    return ParallelEnvironment(env_class=AtariEnvironment, name="MsPacman-ram-v0")


environments = [parallel_cartpole, parallel_pacman]


@pytest.fixture(params=environments, scope="class")
def env(request) -> ClassicControl:
    return request.param()
