"""Various environments for plangym."""
import warnings

try:
    from plangym.atari import AtariEnvironment
    from plangym.classic_control import ClassicControl
    from plangym.dm_control import DMControlEnv, ParallelDMControl
    from plangym.minimal import MinimalPacman, MinimalPong
    from plangym.montezuma import Montezuma
    from plangym.parallel import ParallelEnvironment

except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Failed to import module in plangym.__init__.py: %s" % str(e))
from plangym.version import __version__
