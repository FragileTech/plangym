import os
import warnings

import gym
import pytest

from plangym.control.classic_control import ClassicControl
from plangym.environment_names import ATARI, BOX_2D, CLASSIC_CONTROL, DM_CONTROL, RETRO
from plangym.registry import make
from plangym.vectorization.parallel import ParallelEnv
from tests import (
    SKIP_ATARI_TESTS,
    SKIP_BOX2D_TESTS,
    SKIP_DM_CONTROL_TESTS,
    SKIP_RAY_TESTS,
    SKIP_RETRO_TESTS,
)


def _test_env_class(name, cls, **kwargs):
    n_workers = 2
    assert isinstance(make(name, delay_setup=False, **kwargs), cls)
    env = make(name=name, n_workers=n_workers, delay_setup=True, **kwargs)
    assert isinstance(env, ParallelEnv)
    assert env._env_class == cls
    assert env.n_workers == n_workers
    if not SKIP_RAY_TESTS:
        from plangym.vectorization.ray import RayEnv

        env = make(name=name, n_workers=n_workers, ray=True, delay_setup=True, **kwargs)
        assert isinstance(env, RayEnv)
        assert env._env_class == cls
        assert env.n_workers == n_workers


class TestMake:
    @pytest.mark.parametrize("name", CLASSIC_CONTROL)
    def test_classic_control_make(self, name):
        _test_env_class(name, ClassicControl)

    @pytest.mark.skipif(SKIP_ATARI_TESTS, reason="Atari not installed")
    @pytest.mark.parametrize("name", ATARI[::10])
    def test_atari_make(self, name):
        from plangym.videogames.atari import AtariEnv

        _test_env_class(name, AtariEnv)

    @pytest.mark.skipif(SKIP_BOX2D_TESTS, reason="BOX_2D not installed")
    @pytest.mark.parametrize("name", BOX_2D)
    def test_box2d_make(self, name):
        from plangym.control.box_2d import Box2DEnv
        from plangym.control.lunar_lander import LunarLander

        if name == "FastLunarLander-v0":
            _test_env_class(name, LunarLander)
            return
        elif name == "CarRacing-v0" and os.getenv("SKIP_RENDER", False):
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _test_env_class(name, Box2DEnv)

    @pytest.mark.skipif(SKIP_RETRO_TESTS, reason="Retro not installed")
    @pytest.mark.parametrize("name", RETRO[::10])
    def test_retro_make(self, name):
        from plangym.videogames.retro import RetroEnv

        try:
            _test_env_class(name, RetroEnv)
        except FileNotFoundError:
            pass

    @pytest.mark.skipif(SKIP_RETRO_TESTS, reason="Retro not installed")
    def test_retro_make_with_state(self):
        from plangym.videogames.retro import ActionDiscretizer, RetroEnv

        try:
            _test_env_class(
                "SonicTheHedgehog-Genesis",
                RetroEnv,
                state="GreenHillZone.Act3",
                wrappers=[ActionDiscretizer],
            )
        except FileNotFoundError:
            pass

    @pytest.mark.skipif(SKIP_ATARI_TESTS, reason="Atari not installed")
    def test_custom_atari_make(self):
        # from plangym.minimal import MinimalPacman, MinimalPong
        from plangym.videogames import MontezumaEnv

        # _test_env_class("MinimalPacman-v0", MinimalPacman)
        #  _test_env_class("MinimalPong-v0", MinimalPong)
        _test_env_class("PlanMontezuma-v0", MontezumaEnv)

    @pytest.mark.skipif(SKIP_DM_CONTROL_TESTS, reason="dm_control not installed")
    @pytest.mark.parametrize("name", DM_CONTROL)
    def test_dmcontrol_make(self, name):
        from plangym.control.dm_control import DMControlEnv

        domain_name, task_name = name
        if task_name is not None:
            _test_env_class(domain_name, DMControlEnv, task_name=task_name)
        else:
            _test_env_class(domain_name, DMControlEnv)

    @pytest.mark.skipif(SKIP_DM_CONTROL_TESTS, reason="dm_control not installed")
    @pytest.mark.parametrize("name", DM_CONTROL)
    def test_dmcontrol_domain_name_make(self, name):
        from plangym.control.dm_control import DMControlEnv

        domain_name, task_name = name
        if task_name is not None:
            _test_env_class(
                name=None, domain_name=domain_name, cls=DMControlEnv, task_name=task_name
            )
        else:
            _test_env_class(name=None, domain_name=domain_name, cls=DMControlEnv)

    def test_invalid_name(self):
        with pytest.raises(gym.error.Error):
            make(name="Miaudb")
        with pytest.raises(gym.error.UnregisteredEnv):
            make(name="Miaudb-v0")
