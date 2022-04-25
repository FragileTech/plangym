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
from plangym.environment_names import BOX_2D


def bipedal_walker():
    timelimit = [(TimeLimit, {"max_episode_steps": 1000})]
    return Box2DEnv(name="BipedalWalker-v3", autoreset=True, wrappers=timelimit)


@pytest.fixture(
    params=generate_test_cases(BOX_2D, Box2DEnv, custom_tests=[bipedal_walker]), scope="module"
)
def env(request) -> Box2DEnv:
    return request.param()
