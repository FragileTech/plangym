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
    "ignore",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence",
)

warnings.filterwarnings("ignore", message="Gdk.Cursor.new is deprecated")
warnings.filterwarnings(
    "ignore",
    message=(
        " `numpy.bool` is a deprecated alias for the builtin `bool`. "
        "To silence this warning, use `bool` by itself. Doing this will not modify any "
        "behavior and is safe. If you specifically wanted the numpy scalar type, "
        "use `numpy.bool_` here."
    ),
)
warnings.filterwarnings(
    "ignore",
    message=" WARN: Box bound precision lowered by casting to float32",
)
warnings.filterwarnings(
    "ignore",
    message=(
        " DeprecationWarning: The binary mode of fromstring is deprecated, "
        "as it behaves surprisingly on unicode inputs. Use frombuffer instead"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        " DeprecationWarning: distutils Version classes are deprecated. "
        "Use packaging.version instead."
    ),
)
warnings.filterwarnings(
    "ignore",
    message=" WARNING:root:The use of `check_types` is deprecated and does not have any effect.",
)

from plangym.version import __version__  # noqa: E402


try:
    from plangym.control import Box2DEnv, ClassicControl, DMControlEnv, LunarLander  # noqa: E402
    from plangym.core import PlanEnv  # noqa: E402
    from plangym.registry import make  # noqa: E402
    from plangym.vectorization import ParallelEnv, RayEnv  # noqa: E402
    from plangym.videogames import AtariEnv, MarioEnv, MontezumaEnv, RetroEnv  # noqa: E402
except Exception:
    pass
