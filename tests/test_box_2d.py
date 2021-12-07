from typing import Union

import pytest

from plangym.box_2d import Box2DEnv, LunarLander
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def bipedal_walker():
    return Box2DEnv(name="BipedalWalker-v3", autoreset=True)


def lunar_lander_plangym():
    return LunarLander(autoreset=False)


environments = [bipedal_walker, lunar_lander_plangym]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Union[Box2DEnv, LunarLander]:
    return request.param()
