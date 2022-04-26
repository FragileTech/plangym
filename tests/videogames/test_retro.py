from typing import Union

import gym
import pytest

from plangym.vectorization.parallel import ParallelEnv
from plangym.videogames.retro import ActionDiscretizer, RetroEnv


pytest.importorskip("retro")
from plangym.api_tests import batch_size, display, TestPlanEnv, TestPlangymEnv  # noqa: F401


def retro_airstrike():
    res_obs = gym.wrappers.resize_observation.ResizeObservation
    return RetroEnv(name="Airstriker-Genesis", wrappers=[(res_obs, {"shape": (90, 90)})])


def retro_sonic():

    return RetroEnv(
        name="SonicTheHedgehog-Genesis",
        state="GreenHillZone.Act3",
        wrappers=[ActionDiscretizer],
        obs_type="grayscale",
    )


def parallel_retro():
    return ParallelEnv(
        name="Airstriker-Genesis",
        env_class=RetroEnv,
        n_workers=2,
        obs_type="ram",
        wrappers=[ActionDiscretizer],
    )


environments = [retro_airstrike, retro_sonic, parallel_retro]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Union[RetroEnv, ParallelEnv]:
    env_ = request.param()
    if env_.delay_setup and env_.gym_env is None:
        env_.setup()
    yield env_
    env_.close()


class TestRetro:
    def test_init_env(self):
        env = retro_airstrike()
        env.reset()
        env.setup()

    def test_getattribute(self):
        env = retro_airstrike()
        env.em.get_state()

    def test_clone(self):
        env = RetroEnv(name="Airstriker-Genesis", obs_type="ram", delay_setup=True)
        new_env = env.clone()
        del env
        new_env.reset()
