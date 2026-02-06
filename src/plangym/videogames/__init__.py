"""Module that contains environments representing video games.

Imports are lazy so that missing optional dependencies (e.g. ale-py, stable-retro)
do not prevent importing unrelated environment classes.

.. note::
    **Python Version Compatibility**:

    - ``RetroEnv``: Requires ``stable-retro``, only available on Python 3.10.
"""

import importlib

from plangym import warn_import_error

_SUBMODULES = {
    "AtariEnv": "plangym.videogames.atari",
    "MontezumaEnv": "plangym.videogames.montezuma",
    "MarioEnv": "plangym.videogames.nes",
    "RetroEnv": "plangym.videogames.retro",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        try:
            module = importlib.import_module(_SUBMODULES[name])
            return getattr(module, name)
        except ImportError:
            if name == "RetroEnv":
                warn_import_error("RetroEnv", "stable-retro is only available on Python 3.10.")
                return None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
