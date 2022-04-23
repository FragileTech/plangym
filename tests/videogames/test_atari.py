from gym.wrappers import TimeLimit
import numpy
import numpy as np
import pytest

from plangym.videogames.atari import ale_to_ram, AtariEnv
from tests import SKIP_ATARI_TESTS


if SKIP_ATARI_TESTS:
    pytest.skip("Atari not installed, skipping", allow_module_level=True)
from plangym.api_tests import batch_size, display, TestPlanEnvironment, TestPlangymEnv


def pacman_obs():
    return AtariEnv(name="MsPacman-v0", clone_seeds=True, autoreset=True)


def qbert_ram():
    return AtariEnv(name="Qbert-ram-v0", clone_seeds=False, autoreset=False)


def pong_obs_ram():
    timelimit = [(TimeLimit, {"max_episode_steps": 1000})]
    return AtariEnv(
        name="PongDeterministic-v4",
        remove_time_limit=True,
        possible_to_win=True,
        wrappers=timelimit,
    )


def qbert_new_ale():
    return AtariEnv(name="ALE/Qbert-v5")


environments = [pacman_obs, qbert_ram, pong_obs_ram, qbert_new_ale]


@pytest.fixture(params=environments, scope="class")
def env(request) -> AtariEnv:
    return request.param()


class TestAtariEnv:
    def test_ale_to_ram(self, env):
        ram = ale_to_ram(env.ale)
        assert isinstance(ram, numpy.ndarray)
        assert (ram == env.get_ram()).all()

    def test_get_image(self):
        env = pacman_obs()
        obs = env.get_image()
        assert isinstance(obs, np.ndarray)

    def test_n_actions(self, env):
        n_actions = env.n_actions
        assert isinstance(n_actions, int)
