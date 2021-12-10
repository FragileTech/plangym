from typing import Union

import pytest

from plangym.parallel import ParallelEnvironment
from plangym.retro import RetroEnvironment, SonicDiscretizer


pytest.importorskip("retro")
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def retro_airstrike():
    return RetroEnvironment(name="Airstriker-Genesis", obs_type="ram")


def retro_sonic():

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


environments = [retro_airstrike, retro_sonic, parallel_retro]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Union[RetroEnvironment, ParallelEnvironment]:
    env_ = request.param()
    if env_.delay_init and env_.gym_env is None:
        env_.init_env()
    yield env_
    env_.close()


class TestRetro:
    def test_init_env(self):
        env = retro_airstrike()
        env.reset()
        env.init_env()

    def test_getattribute(self):
        env = retro_airstrike()
        env.em.get_state()

    def test_clone(self):
        env = RetroEnvironment(name="Airstriker-Genesis", obs_type="ram", delay_init=True)
        new_env = env.clone()
        del env
        new_env.reset()
