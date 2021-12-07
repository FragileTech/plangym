import warnings

import pytest
import ray

from plangym.atari import AtariEnvironment
from plangym.classic_control import ClassicControl
from plangym.ray import RayEnv
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def ray_cartpole():
    return RayEnv(env_class=ClassicControl, name="CartPole-v0", n_workers=2)


def ray_retro():
    from plangym.retro import RetroEnvironment

    return RayEnv(env_class=RetroEnvironment, name="Airstriker-Genesis", n_workers=2)


def ray_dm_control():
    from plangym.dm_control import DMControlEnv

    return RayEnv(env_class=DMControlEnv, name="walker-walk", n_workers=2)


environments = [ray_cartpole, ray_retro, ray_dm_control]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ray.init(ignore_reinit_error=True)


@pytest.fixture(params=environments, scope="class")
def env(request) -> AtariEnvironment:
    return request.param()
