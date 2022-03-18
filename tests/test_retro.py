from typing import Union

import pytest

from plangym.parallel import ParallelEnvironment
from plangym.retro import Downsample, RetroEnvironment, SonicDiscretizer


pytest.importorskip("retro")
from plangym.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def retro_airstrike():
    return RetroEnvironment(name="Airstriker-Genesis", wrappers=[(Downsample, {"ratio": 2})])


def retro_sonic():

    return RetroEnvironment(
        name="SonicTheHedgehog-Genesis",
        state="GreenHillZone.Act3",
        wrappers=[SonicDiscretizer],
        obs_type="grayscale",
    )


def parallel_retro():
    return ParallelEnvironment(
        name="Airstriker-Genesis",
        env_class=RetroEnvironment,
        n_workers=2,
        obs_type="ram",
        wrappers=[SonicDiscretizer],
    )


environments = [retro_airstrike, retro_sonic, parallel_retro]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Union[RetroEnvironment, ParallelEnvironment]:
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
        env = RetroEnvironment(name="Airstriker-Genesis", obs_type="ram", delay_setup=True)
        new_env = env.clone()
        del env
        new_env.reset()
