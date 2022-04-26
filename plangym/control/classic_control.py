"""Implement the ``plangym`` API for ``gym`` classic control environments."""
import copy

import numpy

from plangym.core import PlangymEnv


class ClassicControl(PlangymEnv):
    """Environment for OpenAI gym classic control environments."""

    def get_state(self) -> numpy.ndarray:
        """Recover the internal state of the environment."""
        return numpy.array(copy.copy(self.gym_env.unwrapped.state))

    def set_state(self, state: numpy.ndarray):
        """
        Set the internal state of the environemnt.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        self.gym_env.unwrapped.state = copy.copy(tuple(state.tolist()))
        return state
