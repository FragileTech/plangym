from gym.wrappers import TimeLimit
import numpy
import pytest

from plangym.environment_names import ATARI
from plangym.videogames.atari import ale_to_ram, AtariEnv
from tests import SKIP_ATARI_TESTS


if SKIP_ATARI_TESTS:
    pytest.skip("Atari not installed, skipping", allow_module_level=True)
from plangym.api_tests import (  # noqa: F401
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)


def qbert_ram():
    return AtariEnv(name="Qbert-ram-v4", clone_seeds=False, autoreset=False)


@pytest.fixture(
    params=generate_test_cases(ATARI, AtariEnv, custom_tests=[qbert_ram]),
    scope="module",
)
def env(request) -> AtariEnv:
    env = request.param()
    yield env
    env.close()


class TestAtariEnv:
    def test_ale_to_ram(self, env):
        ram = ale_to_ram(env.ale)
        assert isinstance(ram, numpy.ndarray)
        assert (ram == env.get_ram()).all()

    def test_get_image(self):
        env = qbert_ram()
        obs = env.get_image()
        assert isinstance(obs, numpy.ndarray)

    def test_n_actions(self, env):
        n_actions = env.n_actions
        assert isinstance(n_actions, int)
