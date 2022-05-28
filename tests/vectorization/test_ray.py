import os
import warnings

import numpy
import pytest
import ray

from plangym.control.classic_control import ClassicControl
from plangym.vectorization.ray import RayEnv, RemoteEnv
from plangym.videogames.atari import AtariEnv


pytest.importorskip("ray")
if os.getenv("DISABLE_RAY", False) and str(os.getenv("DISABLE_RAY", "False")).lower() != "false":
    pytest.skip("Ray not installed or disabled", allow_module_level=True)
from plangym.api_tests import batch_size, display, TestPlanEnv, TestPlangymEnv  # noqa: F401


def ray_cartpole():
    return RayEnv(env_class=ClassicControl, name="CartPole-v0", n_workers=2)


def ray_retro():
    from plangym.videogames.retro import RetroEnv

    return RayEnv(env_class=RetroEnv, name="Airstriker-Genesis", n_workers=2)


def ray_dm_control():
    from plangym.control.dm_control import DMControlEnv

    return RayEnv(env_class=DMControlEnv, name="walker-walk", n_workers=2)


environments = [(ray_cartpole, True), (ray_retro, False), (ray_dm_control, True)]


@pytest.fixture(params=environments, scope="module")
def env(request) -> AtariEnv:
    env_call, local = request.param
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ray.init(ignore_reinit_error=True, local_mode=local)
    yield env_call()
    ray.shutdown()


def test_remote_actor():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ray.init(ignore_reinit_error=True, local_mode=True)

    def create_cartpole():
        return ClassicControl(name="CartPole-v0")

    env = RemoteEnv.remote(create_cartpole)
    ray.get(env.setup.remote())
    ray.get(env.reset.remote())
    ray.get(env.step.remote(0))
    state = ray.get(env.get_state.remote())
    assert isinstance(state, numpy.ndarray)
