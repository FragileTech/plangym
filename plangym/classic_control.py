"""Implement the ``plangym`` API for ``gym`` classic control environments."""
import copy

import numpy as np

from plangym.core import PlanEnvironment


class ClassicControl(PlanEnvironment):
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

    def reset(self, return_state: bool = True):
        """
        Restart the environment.

        Args:
            return_state: If ``True`` it will return the state of the environment.

        Returns:
            ``obs`` if ```return_state`` is ``True`` else return ``(state, obs)``.

        """
        if not return_state:
            return self.gym_env.reset()
        else:
            obs = self.gym_env.reset()
            return self.get_state(), obs
