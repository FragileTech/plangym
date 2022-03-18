from typing import Union

from gym.wrappers import TimeLimit
import pytest


pytest.importorskip("Box2D")
from plangym.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment
from plangym.box_2d import Box2DEnv, LunarLander


def bipedal_walker():
    timelimit = [(TimeLimit, {"max_episode_steps": 1000})]
    return Box2DEnv(name="BipedalWalker-v3", autoreset=True, wrappers=timelimit)


def lunar_lander_det_discrete():
    return LunarLander(autoreset=False, deterministic=True, continuous=False)


def lunar_lander_random_discrete():
    return LunarLander(autoreset=False, deterministic=False, continuous=False)


def lunar_lander_random_continuous():
    return LunarLander(
        autoreset=False,
        deterministic=False,
        continuous=True,
    )


environments = [
    bipedal_walker,
    lunar_lander_det_discrete,
    lunar_lander_random_discrete,
    lunar_lander_random_continuous,
]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Union[Box2DEnv, LunarLander]:
    return request.param()
