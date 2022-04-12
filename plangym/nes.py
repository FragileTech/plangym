"""Environment for playing Mario bros using gym-super-mario-bros."""
import gym
import numpy as np

from plangym.core import PlanEnvironment


class MarioEnvironment(PlanEnvironment):
    """Interface for using gym-super-mario-bros in plangym."""

    def get_image(self) -> np.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        return self.gym_env.screen

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`gym.Env`` instance that the current class is wrapping."""
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        env = gym_super_mario_bros.make(self.name)
        gym_env = JoypadSpace(env.unwrapped, SIMPLE_MOVEMENT)
        gym_env.reset()
        return gym_env

    def get_state(self) -> np.ndarray:
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        return self.gym_env.get_state()

    def set_state(self, state: np.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        self.gym_env.set_state(state)

    def close(self) -> None:
        """Close the underlying :class:`gym.Env`."""
        try:
            super(MarioEnvironment, self).close()
        except ValueError:
            pass
