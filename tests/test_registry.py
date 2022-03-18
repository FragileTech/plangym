import os
import warnings

import pytest

from plangym.classic_control import ClassicControl
from plangym.environment_names import ATARI, BOX_2D, CLASSIC_CONTROL, DM_CONTROL, RETRO
from plangym.parallel import ParallelEnvironment
from plangym.registry import make
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
    assert isinstance(env, ParallelEnvironment)
    assert env._env_class == cls
    assert env.n_workers == n_workers
    if not SKIP_RAY_TESTS:
        from plangym.ray import RayEnv

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
        from plangym.atari import AtariEnvironment

        _test_env_class(name, AtariEnvironment)

    @pytest.mark.skipif(SKIP_BOX2D_TESTS, reason="BOX_2D not installed")
    @pytest.mark.parametrize("name", BOX_2D)
    def test_box2d_make(self, name):
        from plangym.box_2d import Box2DEnv, LunarLander

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
        from plangym.retro import RetroEnvironment

        try:
            _test_env_class(name, RetroEnvironment)
        except FileNotFoundError:
            pass

    @pytest.mark.skipif(SKIP_RETRO_TESTS, reason="Retro not installed")
    def test_retro_make_with_state(self):
        from plangym.retro import RetroEnvironment, SonicDiscretizer

        try:
            _test_env_class(
                "SonicTheHedgehog-Genesis",
                RetroEnvironment,
                state="GreenHillZone.Act3",
                wrappers=[SonicDiscretizer],
            )
        except FileNotFoundError:
            pass

    @pytest.mark.skipif(SKIP_ATARI_TESTS, reason="Atari not installed")
    def test_custom_atari_make(self):
        from plangym.minimal import MinimalPacman, MinimalPong
        from plangym.montezuma import Montezuma

        _test_env_class("MinimalPacman-v0", MinimalPacman)
        _test_env_class("MinimalPong-v0", MinimalPong)
        _test_env_class("PlanMontezuma-v0", Montezuma)

    @pytest.mark.skipif(SKIP_DM_CONTROL_TESTS, reason="dm_control not installed")
    @pytest.mark.parametrize("name", DM_CONTROL)
    def test_dmcontrol_make(self, name):
        from plangym.dm_control import DMControlEnv

        domain_name, task_name = name
        if task_name is not None:
            _test_env_class(domain_name, DMControlEnv, task_name=task_name)
        else:
            _test_env_class(domain_name, DMControlEnv)

    @pytest.mark.skipif(SKIP_DM_CONTROL_TESTS, reason="dm_control not installed")
    @pytest.mark.parametrize("name", DM_CONTROL)
    def test_dmcontrol_domain_name_make(self, name):
        from plangym.dm_control import DMControlEnv

        domain_name, task_name = name
        if task_name is not None:
            _test_env_class(
                name=None, domain_name=domain_name, cls=DMControlEnv, task_name=task_name
            )
        else:
            _test_env_class(name=None, domain_name=domain_name, cls=DMControlEnv)

    def test_invalid_name(self):
        with pytest.raises(ValueError):
            make(name="Miaudb")
