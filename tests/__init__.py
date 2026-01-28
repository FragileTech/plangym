import os
import warnings

import pytest


warnings.filterwarnings(
    action="ignore", category=pytest.PytestUnraisableExceptionWarning, module="pytest"
)

# Disable Ray's uv runtime_env hook early so it's effective before any ray import.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_RUNTIME_ENV_CREATE_WORKING_DIR", "0")
try:
    import retro
    from stable_retro import data as retro_data

    _REQUIRED_RETRO_GAMES = ("Airstriker-Genesis", "SonicTheHedgehog-Genesis")
    SKIP_RETRO_TESTS = False
    for _game in _REQUIRED_RETRO_GAMES:
        try:
            retro_data.get_romfile_path(_game)
        except FileNotFoundError:
            SKIP_RETRO_TESTS = True
            break
except Exception:
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
