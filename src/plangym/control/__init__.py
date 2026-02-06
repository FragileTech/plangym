"""Module that contains environments representing control tasks.

Imports are lazy so that missing optional dependencies (e.g. mujoco, dm-control)
do not prevent importing unrelated environment classes.
"""

import importlib

_SUBMODULES = {
    "BalloonEnv": "plangym.control.balloon",
    "Box2DEnv": "plangym.control.box_2d",
    "ClassicControl": "plangym.control.classic_control",
    "DMControlEnv": "plangym.control.dm_control",
    "LunarLander": "plangym.control.lunar_lander",
    "MujocoEnv": "plangym.control.mujoco",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(_SUBMODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
