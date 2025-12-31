import warnings

import pytest


warnings.filterwarnings(
    action="ignore", category=pytest.PytestUnraisableExceptionWarning, module="pytest"
)
try:
    import retro

    SKIP_RETRO_TESTS = False
except ImportError:
    SKIP_RETRO_TESTS = True

try:
    import ray

    SKIP_RAY_TESTS = False
except ImportError:
    SKIP_RAY_TESTS = True

try:
    from plangym.videogames.atari import AtariEnv

    SKIP_ATARI_TESTS = False
except ImportError:
    SKIP_ATARI_TESTS = True

try:
    from plangym.control.dm_control import DMControlEnv

    DMControlEnv(name="walker-run", frameskip=3)
    SKIP_DM_CONTROL_TESTS = False
except (ImportError, AttributeError, ValueError):
    SKIP_DM_CONTROL_TESTS = True


try:
    import Box2D

    SKIP_BOX2D_TESTS = False
except ImportError:
    SKIP_BOX2D_TESTS = True

try:
    import gym_super_mario_bros

    SKIP_MARIO_TESTS = False
except (ImportError, AttributeError):
    # AttributeError: nes_py depends on old gym which fails on Python 3.12+
    SKIP_MARIO_TESTS = True
