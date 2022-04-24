from typing import Union

from gym.wrappers import TimeLimit
import pytest


pytest.importorskip("Box2D")
from plangym.api_tests import (  # noqa: F401
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)
from plangym.control.box_2d import Box2DEnv
from plangym.control.lunar_lander import LunarLander
from plangym.environment_names import BOX_2D


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

try:
    bipedal_walker().get_image()
except Exception:
    pytest.skip(allow_module_level=True)


@pytest.fixture(params=generate_test_cases(BOX_2D, Box2DEnv), scope="module")
def env(request) -> Union[Box2DEnv, LunarLander]:
    return request.param()
