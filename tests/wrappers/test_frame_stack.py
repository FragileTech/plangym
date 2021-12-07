import pytest


pytest.importorskip("atari_py")

import gym
import numpy as np

from plangym.wrappers import FrameStack


try:
    import lz4
except ImportError:
    lz4 = None


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1", "Pong-v0"])
@pytest.mark.parametrize("num_stack", [2, 3, 4])
@pytest.mark.parametrize(
    "lz4_compress",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(lz4 is None, reason="Need lz4 to run tests with compression"),
        ),
        False,
    ],
)
def test_frame_stack(env_id, num_stack, lz4_compress):
    env = gym.make(env_id)
    shape = env.observation_space.shape
    env = FrameStack(env, num_stack, lz4_compress)
    assert env.observation_space.shape == (num_stack,) + shape

    obs = env.reset()
    obs = np.asarray(obs)
    assert np.prod(obs.shape) == np.prod((num_stack,) + shape)

    obs, _, _, _ = env.step(env.action_space.sample())
    obs = np.asarray(obs)
    assert np.prod(obs.shape) == np.prod((num_stack,) + shape)
