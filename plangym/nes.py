"""Environment for playing Mario bros using gym-super-mario-bros."""
from typing import Any, Dict, Optional

import gym
import numpy
import numpy as np

from plangym.core import VideogameEnvironment


class NesEnvironment(VideogameEnvironment):
    """Environment for working with the NES-py emulator."""

    @property
    def nes_env(self) -> "NESEnv":  # noqa: F821
        """Access the underlying NESEnv."""
        return self.gym_env.unwrapped

    def get_image(self) -> np.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        return self.gym_env.screen.copy()

    def get_ram(self) -> np.ndarray:
        """Return a copy of the emulator environment."""
        return self.nes_env.ram.copy()

    def get_state(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        return self.gym_env.get_state(state)

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
        if self.nes_env._env is None:
            return
        super(NesEnvironment, self).close()


class MarioEnvironment(NesEnvironment):
    """Interface for using gym-super-mario-bros in plangym."""

    def get_state(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        state = np.empty(250288, dtype=np.byte) if state is None else state
        state[-2:] = 0  # Some states use the last two bytes. Set to zero by default.
        return super(MarioEnvironment, self).get_state(state)

    def init_spaces(self) -> None:
        """Initialize the target :class:`NESEnv` instance."""
        super(MarioEnvironment, self).init_spaces()
        if self.obs_type == "info":
            self._obs_space = gym.spaces.Box(low=0, high=np.inf, dtype=numpy.float32, shape=7)

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`NESEnv`` instance that the current class is wrapping."""
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT  # , SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        env = gym_super_mario_bros.make(self.name)
        gym_env = JoypadSpace(env.unwrapped, COMPLEX_MOVEMENT)
        gym_env.reset()
        return gym_env

    def _update_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info["player_state"] = self.nes_env._player_state
        info["area"] = self.nes_env._area
        info["left_x_position"] = self.nes_env._left_x_position
        info["is_stage_over"] = self.nes_env._is_stage_over
        info["is_dying"] = self.nes_env._is_dying
        info["is_dead"] = self.nes_env._is_dead
        info["y_pixel"] = self.nes_env._y_pixel
        info["y_viewport"] = self.nes_env._y_viewport
        info["x_position_last"] = self.nes_env._x_position_last
        info["in_pipe"] = (info["player_state"] == 0x02) or (info["player_state"] == 0x03)
        return info

    def _get_info(
        self,
    ):
        info = {
            "x_pos": 0,
            "y_pos": 0,
            "world": 0,
            "stage": 0,
            "life": 0,
            "coins": 0,
            "flag_get": False,
            "in_pipe": False,
        }
        return self._update_info(info)

    def process_obs(
        self,
        obs: numpy.ndarray,
        info: Dict[str, Any] = None,
        **kwargs,
    ) -> numpy.ndarray:
        """Return the information contained in info as an observation if obs_type == "info"."""
        if self.obs_type == "info":
            info = info or self._get_info()
            obs = np.array(
                [
                    info.get("x_pos", 0),
                    info.get("y_pos", 0),
                    info.get("world" * 10, 0),
                    info.get("stage", 0),
                    info.get("life", 0),
                    int(info.get("flag_get", 0)),
                    info.get("coins", 0),
                ],
            )
        return obs

    def process_reward(self, reward, info, **kwargs) -> float:
        """Return a custom reward based on the x, y coordinates and level mario is in."""
        reward = (
            (info.get("world", 0) * 25000)
            + (info.get("stage", 0) * 5000)
            + info.get("x_pos", 0)
            + 10 * int(bool(info.get("in_pipe", 0)))
            + 100 * int(bool(info.get("flag_get", 0)))
            # + (abs(info["x_pos"] - info["x_position_last"]))
        )
        return reward

    def process_terminal(self, terminal, info, **kwargs) -> bool:
        """Return True if terminal or mario is dying."""
        return terminal or info.get("is_dying", False) or info.get("is_dead", False)

    def process_info(self, info, **kwargs) -> Dict[str, Any]:
        """Add additional data to the info dictionary."""
        return self._update_info(info)
