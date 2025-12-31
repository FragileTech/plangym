import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TimeLimit, TransformReward
import numpy
from numpy.random import default_rng

from plangym.utils import process_frame, remove_time_limit, process_frame_opencv

rng = default_rng()


def test_remove_time_limit():
    env = gym.make("MsPacmanNoFrameskip-v4")
    env = TransformReward(TimeLimit(AtariPreprocessing(env), max_episode_steps=100), lambda x: x)
    rem_env = remove_time_limit(env)
    assert rem_env.spec.max_episode_steps == int(1e100)
    assert not isinstance(rem_env.env, TimeLimit)
    assert "TimeLimit" not in str(rem_env)


def test_process_frame():
    example = (rng.random((100, 100, 3)) * 255).astype(numpy.uint8)
    frame = process_frame(example, mode="L")
    assert frame.shape == (100, 100)
    frame = process_frame(example, width=30, height=50)
    assert frame.shape == (50, 30, 3)
    frame = process_frame(example, width=80, height=70, mode="L")
    assert frame.shape == (70, 80)


def test_process_frame_opencv():
    example = (rng.random((100, 100, 3)) * 255).astype(numpy.uint8)
    frame = process_frame_opencv(example, mode="L")
    assert frame.shape == (100, 100)
    frame = process_frame_opencv(example, width=30, height=50)
    assert frame.shape == (50, 30, 3)
    frame = process_frame_opencv(example, width=30, height=50, mode="BGR")
    assert frame.shape == (50, 30, 3)
    frame = process_frame_opencv(example, width=80, height=70, mode="GRAY")
    assert frame.shape == (70, 80)
