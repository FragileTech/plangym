import os

import numpy
import pytest


pytest.importorskip("dm_control")
from plangym.dm_control import DMControlEnv
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def walker_run():
    return DMControlEnv(name="walker-run", frameskip=3)


def parallel_dm():
    return DMControlEnv(name="cartpole-balance", frameskip=3)


environments = [walker_run, parallel_dm]


@pytest.fixture(params=environments, scope="class")
def env(request) -> DMControlEnv:
    return request.param()


class TestDMControl:
    def test_attributes(self, env):
        env.reset()
        assert hasattr(env, "physics")
        assert hasattr(env, "action_spec")
        assert hasattr(env, "action_space")
        assert hasattr(env, "render_mode")
        assert env.render_mode in {"human", "rgb_array", None}

    @pytest.mark.skipif(os.getenv("SKIP_RENDER", False), reason="No display in CI.")
    def test_render(self, env):
        env.reset()
        obs_rgb = env.render(mode="rgb_array")
        assert isinstance(obs_rgb, numpy.ndarray)
        old_len = len(env.viewer)
        action = env.sample_action()
        env.step(action)
        env.render(mode="human")
        assert len(env.viewer) > old_len
        env.show_game(sleep=0.01)

    def test_parse_name_fails(self):
        with pytest.raises(ValueError):
            DMControlEnv(name="cartpole")
