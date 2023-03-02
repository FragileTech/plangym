"""Implement the ``plangym`` API for the Balloon Learning Environment."""
from typing import Any

import numpy


try:
    import balloon_learning_environment.env.balloon_env  # noqa: F401
    from balloon_learning_environment.env.rendering.matplotlib_renderer import MatplotlibRenderer
except ImportError:

    def MatplotlibRenderer():  # noqa: D103
        return None


from plangym.core import PlangymEnv


class BalloonEnv(PlangymEnv):
    """
    This class implements the 'BalloonLearningEnvironment-v0' released by Google in the \
    balloon_learning_environment.

    For more information about the environment, please refer to \
    https://github.com/google/balloon-learning-environment.
    """

    AVAILABLE_RENDER_MODES = {"human", "rgb_array", "tensorboard", None}
    AVAILABLE_OBS_TYPES = {"coords", "rgb", "grayscale"}
    STATE_IS_ARRAY = False

    def __init__(
        self,
        name: str = "BalloonLearningEnvironment-v0",
        renderer=None,
        array_state: bool = True,
        **kwargs,
    ):
        """
        Initialize a :class:`BalloonEnv`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            renderer: MatplotlibRenderer object (or any renderer object) to plot
                the ``balloons`` environment. For more information, see the
                official documentation.
            array_state: boolean value. If True, transform the state object to
                a ``numpy.array``.
        """
        renderer = renderer or MatplotlibRenderer()
        self.STATE_IS_ARRAY = array_state
        super(BalloonEnv, self).__init__(name=name, renderer=renderer, **kwargs)

    def get_state(self) -> Any:
        """Get the state of the environment."""
        state = self.gym_env.unwrapped.get_simulator_state()
        if self.STATE_IS_ARRAY:
            state = numpy.array((state, None), dtype=object)
        return state

    def set_state(self, state: Any) -> None:
        """Set the state of the environment."""
        if self.STATE_IS_ARRAY:
            state = state[0]
        return self.gym_env.unwrapped.arena.set_simulator_state(state)

    def seed(self, seed: int = None):
        """Ignore seeding until next release."""
        return
