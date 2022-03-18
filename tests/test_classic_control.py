from gym.wrappers import TimeLimit
import pytest

from plangym.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment
from plangym.classic_control import ClassicControl


environments = [
    "MountainCar-v0",
    "Acrobot-v1",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "CartPole-v0",
]


@pytest.fixture(params=environments, scope="class")
def env(request) -> ClassicControl:
    name = request.param
    timelimit = [(TimeLimit, {"max_episode_steps": 1000})]
    return ClassicControl(name=name, delay_setup=name == "CartPole-v0", wrappers=timelimit)
