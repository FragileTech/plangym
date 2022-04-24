import pytest

from plangym.api_tests import batch_size, display, generate_test_cases, TestPlanEnv  # noqa: F401
from plangym.vectorization import ParallelEnvironment
from plangym.videogames.nes import MarioEnv


env_names = ["SuperMarioBros-v0", "SuperMarioBros-v1", "SuperMarioBros2-v0"]


@pytest.fixture(
    params=generate_test_cases(env_names, MarioEnv, n_workers_values=[None, 2]), scope="module"
)
def env(request):
    env = request.param()
    yield env
    env.close()
