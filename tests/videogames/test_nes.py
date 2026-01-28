import pytest

from tests import SKIP_MARIO_TESTS

if SKIP_MARIO_TESTS:
    pytest.skip("gym_super_mario_bros not installed, skipping", allow_module_level=True)

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


class TestMarioEnv:
    def test_get_keys_to_action(self, env):
        vals = env.gym_env.get_keys_to_action()
        assert isinstance(vals, dict)

    def test_get_action_meanings(self, env):
        vals = env.gym_env.get_action_meanings()
        assert isinstance(vals, list)

    def test_buttons(self, env):
        buttons = env.gym_env.buttons()
        assert isinstance(buttons, list), buttons
        assert all(isinstance(b, str) for b in buttons)
