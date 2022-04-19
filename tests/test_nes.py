import pytest

from plangym.api_tests import batch_size, display, TestBaseEnvironment
from plangym.nes import MarioEnvironment
from plangym.parallel import ParallelEnvironment


def single_core_mario():
    return MarioEnvironment("SuperMarioBros-v0")


def parallel_mario():
    return ParallelEnvironment(env_class=MarioEnvironment, name="SuperMarioBros-v0", n_workers=2)


environments = [single_core_mario, parallel_mario]


@pytest.fixture(params=environments, scope="class")
def env(request):
    return request.param()
