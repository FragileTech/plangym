"""Implement the ``plangym`` API for ``gym`` classic control environments."""
import copy

import numpy as np

from plangym.core import PlangymEnv


class ClassicControl(PlangymEnv):
    """Environment for OpenAI gym classic control environments."""

    def get_state(self) -> np.ndarray:
        """Recover the internal state of the environment."""
        return np.array(copy.copy(self.gym_env.unwrapped.state))

    def set_state(self, state: np.ndarray):
        """
        Set the internal state of the environemnt.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        self.gym_env.unwrapped.state = copy.copy(tuple(state.tolist()))
        return state
