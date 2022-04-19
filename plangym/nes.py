"""Environment for playing Mario bros using gym-super-mario-bros."""
from typing import Any, Dict, Optional, Tuple, Union

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

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Return the shape of the environment observations."""
        return tuple([7])

    def get_state(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        state = np.empty(250288, dtype=np.byte) if state is None else state
        state[-2:] = 0  # Some states use the last two bytes. Set to zero by default.
        return super(MarioEnvironment, self).get_state(state)

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`gym.Env`` instance that the current class is wrapping."""
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

    def _get_obs(self, obs: numpy.ndarray, info: Dict[str, Any]) -> numpy.ndarray:
        obs = np.array(
            [
                info["x_pos"],
                info["y_pos"],
                info["world"] * 10,
                info["stage"],
                info["life"],
                int(info["flag_get"]),
                info["coins"],
            ],
        )
        return obs

    def step(
        self,
        action: Union[numpy.ndarray, int, float],
        state: numpy.ndarray = None,
        dt: int = 1,
    ) -> tuple:
        """
        Step the environment applying the supplied action.

        Optionally set the state to the supplied state before stepping it.

        Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            if state is None returns ``(observs, reward, terminal, info)``
            else returns ``(new_state, observs, reward, terminal, info)``

        """
        data = super(MarioEnvironment, self).step(action=action, state=state, dt=dt)
        *new_state, obs, reward, terminal, info = data
        info = self._update_info(info)
        obs = self._get_obs(obs, info)
        reward = (
            (info["world"] * 25000)
            + (info["stage"] * 5000)
            + info["x_pos"]
            + 10 * int(bool(info["in_pipe"]))
            + 100 * int(bool(info["flag_get"]))
            # + (abs(info["x_pos"] - info["x_position_last"]))
        )
        terminal = terminal or info["is_dying"] or info["is_dead"]
        ret_data = obs, reward, terminal, info
        return ret_data if state is None else (new_state[0], obs, reward, terminal, info)

    def reset(
        self,
        return_state: bool = True,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Reset the environment and returns the first observation, or the first \
        (state, obs) tuple.

        Args:
            return_state: If true return a also the initial state of the env.

        Returns:
            Observation of the environment if `return_state` is False. Otherwise,
            return (state, obs) after reset.

        """
        data = super(MarioEnvironment, self).reset(return_state=return_state)
        obs = np.zeros(7)
        return (data[0], obs) if return_state else obs
