"""Various environments for plangym."""
import warnings


warnings.filterwarnings(
    "ignore",
    message=(
        "Using or importing the ABCs from 'collections' instead of from 'collections.abc' "
        "is deprecated since Python 3.3,and in 3.9 it will stop working"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "the imp module is deprecated in favour of importlib; see the module's "
        "documentation for alternative uses"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "Using or importing the ABCs from 'collections' instead of from "
        "'collections.abc' is deprecated, and in 3.8 it will stop working"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "The set_clim function was deprecated in Matplotlib 3.1 "
        "and will be removed in 3.3. Use ScalarMappable.set_clim "
        "instead."
    ),
)
warnings.filterwarnings(
    "ignore", category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message="invalid escape sequence",
)

warnings.filterwarnings("ignore", message="Gdk.Cursor.new is deprecated")
try:
    from plangym.atari import AtariEnvironment  # noqa: E402
    from plangym.classic_control import ClassicControl  # noqa: E402
    from plangym.core import BaseEnvironment  # noqa: E402
    from plangym.dm_control import DMControlEnv, ParallelDMControl  # noqa: E402
    from plangym.minimal import MinimalPacman, MinimalPong  # noqa: E402
    from plangym.montezuma import Montezuma  # noqa: E402
    from plangym.parallel import ParallelEnvironment  # noqa: E402
    from plangym.retro import ParallelRetro, RetroEnvironment  # noqa: E402

except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Failed to import module in plangym.__init__.py: %s" % str(e))
from plangym.version import __version__  # noqa: E402
