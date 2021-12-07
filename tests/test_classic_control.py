import pytest

from plangym.classic_control import ClassicControl
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


environments = [
    "MountainCar-v0",
    "Acrobot-v1",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "CartPole-v0",
]


@pytest.fixture(params=environments, scope="class")
def env(request) -> ClassicControl:
    return ClassicControl(name=request.param)
