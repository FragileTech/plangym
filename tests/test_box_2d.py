from typing import Union

import pytest

from plangym.box_2d import Box2DEnv, LunarLander
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def bipedal_walker():
    return Box2DEnv(name="BipedalWalker-v3", autoreset=True)


def lunar_lander_det_discrete():
    return LunarLander(autoreset=False, deterministic=True, continuous=False)


def lunar_lander_random_discrete():
    return LunarLander(autoreset=False, deterministic=False, continuous=False)


def lunar_lander_random_continuous():
    return LunarLander(autoreset=False, deterministic=False, continuous=True)


lunar_lander_envs = [
    lunar_lander_det_discrete,
    lunar_lander_random_discrete,
    lunar_lander_random_continuous,
]

environments = [bipedal_walker] + lunar_lander_envs


@pytest.fixture(params=environments, scope="class")
def env(request) -> Union[Box2DEnv, LunarLander]:
    return request.param()


class TestLunarLander:
    def test_lunar_lander_death(self):
        env = lunar_lander_random_discrete()
        env.reset()
        for i in range(50):
            env.step(env.sample_action())
