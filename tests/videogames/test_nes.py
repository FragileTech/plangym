import pytest

from plangym.api_tests import batch_size, display, TestPlanEnvironment
from plangym.vectorization import ParallelEnvironment
from plangym.videogames.nes import MarioEnv


def single_core_mario():
    return MarioEnv("SuperMarioBros-v0")


def parallel_mario():
    return ParallelEnvironment(env_class=MarioEnv, name="SuperMarioBros-v0", n_workers=2)


environments = [single_core_mario, parallel_mario]


@pytest.fixture(params=environments, scope="class")
def env(request):
    return request.param()
