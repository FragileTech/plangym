"""Module that contains environments representing video games.

.. note::
    **Python Version Compatibility**:

    - ``RetroEnv``: Requires ``stable-retro``, only available on Python 3.10.
"""

from plangym import warn_import_error
from plangym.videogames.atari import AtariEnv
from plangym.videogames.montezuma import MontezumaEnv
from plangym.videogames.nes import MarioEnv

try:
    from plangym.videogames.retro import RetroEnv
except ImportError:
    RetroEnv = None
    warn_import_error("RetroEnv", "stable-retro is only available on Python 3.10.")
