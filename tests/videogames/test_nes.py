import pytest

from plangym.api_tests import (
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
    TestVideogameEnv,
)
from plangym.videogames.nes import MarioEnv


env_names = ["SuperMarioBros-v0", "SuperMarioBros-v1", "SuperMarioBros2-v0"]


@pytest.fixture(
    params=generate_test_cases(env_names, MarioEnv, n_workers_values=None), scope="module"
)
def env(request):
    return request.param()


# class TestMarioEnv(TestPlangymEnv):
#    pass
