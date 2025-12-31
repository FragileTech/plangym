"""Various environments for plangym."""

import logging
import os
import warnings

# Global flag to disable plangym warnings (set via environment variable)
PLANGYM_DISABLE_WARNINGS = os.environ.get("PLANGYM_DISABLE_WARNINGS", "").lower() in {
    "true",
    "1",
    "yes",
}

_logger = logging.getLogger(__name__)


def warn_import_error(module_name: str, reason: str = "") -> None:
    """Log a warning when an optional package fails to import.

    Args:
        module_name: Name of the module that failed to import.
        reason: Optional explanation of why the import failed.

    Note:
        Warnings can be disabled by setting the environment variable
        ``PLANGYM_DISABLE_WARNINGS=true``.

    """
    if not PLANGYM_DISABLE_WARNINGS:
        msg = f"Could not import {module_name}."
        if reason:
            msg += f" {reason}"
        _logger.warning(msg)


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

from plangym.core import PlanEnv
from plangym.registry import make
from plangym.version import __version__
