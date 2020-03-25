"""Various environments for plangym."""
import warnings

try:
    from plangym.dm_control import DMControlEnv, ParallelDMControl
    from plangym.env import AtariEnvironment, ParallelEnvironment
    from plangym.minimal import ClassicControl, MinimalPacman, MinimalPong
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Failed to import module in plangym.__init__.py: %s" % str(e))
from plangym.version import __version__
