from typing import Union

import pytest

from plangym.parallel import ParallelEnvironment
from plangym.retro import RetroEnvironment
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def retro_airstrike():
    return RetroEnvironment(name="Airstriker-Genesis")


def retro_sonic():
    from plangym.wrappers.retro_wrappers import SonicDiscretizer

    return RetroEnvironment(
        name="SonicTheHedgehog-Genesis",
        state="GreenHillZone.Act3",
        wrappers=[SonicDiscretizer],
    )


# retro.make("Airstriker-Genesis")
def parallel_retro():
    return ParallelEnvironment(
        name="Airstriker-Genesis",
        env_class=RetroEnvironment,
        n_workers=2,
        delay_init=False,
    )


environments = [retro_airstrike]  # , retro_sonic, parallel_retro]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Union[RetroEnvironment, ParallelEnvironment]:
    env_ = request.param()
    if env_.delay_init and env_.gym_env is None:
        env_.init_env()
    yield env_
    env_.close()
