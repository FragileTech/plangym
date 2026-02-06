import numpy as np
from gymnasium.wrappers import TimeLimit
import numpy
import pytest

from plangym.environment_names import ATARI
from plangym.videogames.atari import ale_to_ram, AtariEnv
from tests import SKIP_ATARI_TESTS


if SKIP_ATARI_TESTS:
    pytest.skip("Atari not installed, skipping", allow_module_level=True)
from plangym.api_tests import (
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)


def qbert_ram():
    return AtariEnv(name="ALE/Qbert-v5", obs_type="ram", clone_seeds=False, autoreset=False)


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
        _ = env.reset()
        ram = ale_to_ram(env.ale)
        env_ram = env.get_ram()
        assert isinstance(ram, numpy.ndarray)
        assert ram.shape == env_ram.shape
        assert (ram == env_ram).all()

    def test_get_image(self):
        env = qbert_ram()
        obs = env.get_image()
        assert isinstance(obs, numpy.ndarray)

    def test_ram_obs_with_image_extraction(self):
        """Verify that RGB images can be extracted when using obs_type='ram'."""
        env = qbert_ram()
        _state, obs, _info = env.reset()
        # RAM observation should be 1D uint8 array of shape (128,)
        assert obs.ndim == 1
        assert obs.shape == (128,)
        assert obs.dtype == numpy.uint8
        # get_image() should still return a valid RGB image
        img = env.get_image()
        assert img.ndim == 3
        assert img.shape == (210, 160, 3)
        assert img.dtype == numpy.uint8
        env.close()

    def test_ram_obs_return_image(self):
        """Verify return_image=True populates info['rgb'] with obs_type='ram'."""
        env = AtariEnv(
            name="ALE/Qbert-v5", obs_type="ram", return_image=True,
            clone_seeds=False, autoreset=False,
        )
        _state, obs, info = env.reset()
        assert "rgb" in info
        assert info["rgb"].ndim == 3
        assert info["rgb"].shape == (210, 160, 3)
        # Also check step
        obs, _reward, _term, _trunc, info = env.step(env.sample_action())
        assert obs.shape == (128,)
        assert "rgb" in info
        assert info["rgb"].ndim == 3
        assert info["rgb"].shape == (210, 160, 3)
        env.close()

    def test_n_actions(self, env):
        n_actions = env.n_actions
        assert isinstance(n_actions, int | np.int64)
