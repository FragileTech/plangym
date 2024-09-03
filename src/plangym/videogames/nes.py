"""Environment for playing Mario bros using gym-super-mario-bros."""

from typing import Any, TypeVar

import gymnasium as gym
import numpy

from plangym.videogames.env import VideogameEnv

# actions for the simple run right environment
RIGHT_ONLY = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]


# actions for very simple movement
SIMPLE_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
    ["left", "A"],
    ["left", "B"],
    ["left", "A", "B"],
    ["down"],
    ["up"],
]

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class NESWrapper:
    """A wrapper for the NES environment."""

    def __init__(self, wrapped):
        """Initialize the NESWrapper."""
        self._wrapped = wrapped

    def __getattr__(self, name):
        """Get an attribute from the wrapped object."""
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        """Set an attribute on the wrapped object."""
        if name == "_wrapped":
            super().__setattr__(name, value)
        else:
            setattr(self._wrapped, name, value)  # pragma: no cover

    def __delattr__(self, name):
        """Delete an attribute from the wrapped object."""
        delattr(self._wrapped, name)  # pragma: no cover

    def step(
        self, action: ActType
    ) -> tuple[gym.core.WrapperObsType, gym.core.SupportsFloat, bool, bool, dict[str, Any]]:
        """Modify the :attr:`env` after calling :meth:`step` using :meth:`self.observation`."""
        observation, reward, terminated, info = self._wrapped.step(action)
        truncated = False
        return self.observation(observation), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,  # noqa: ARG002
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[gym.core.WrapperObsType, dict[str, Any]]:
        """Modify the :attr:`env` after calling :meth:`reset`, returning a modified observation."""
        obs = self.env.reset()
        info = {}
        return self.observation(obs), info

    def observation(self, observation: ObsType) -> gym.core.WrapperObsType:
        """Return a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation

        """
        return observation


class JoypadSpace(gym.Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        "right": 0b10000000,
        "left": 0b01000000,
        "down": 0b00100000,
        "up": 0b00010000,
        "start": 0b00001000,
        "select": 0b00000100,
        "B": 0b00000010,
        "A": 0b00000001,
        "NOOP": 0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: gym.Env, actions: list):
        """Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(actions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = " ".join(button_list)

    def step(self, action):
        """Take a step using the given action.

        Args:
            action (int): the discrete action to perform

        Returns:
            a tuple of:
            - (numpy.ndarray) the state as a result of the action
            - (float) the reward achieved by taking the action
            - (bool) a flag denoting whether the episode has ended
            - (dict) a dictionary of extra information

        """
        # take the step and record the output
        return self.env.step(self._action_map[action])

    # def reset(self, *, seed=None, options=None):
    #    """Reset the environment and return the initial observation."""
    #    return self.env.reset(), {}

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        # get the old mapping of keys to actions
        old_keys_to_action = self.env.unwrapped.get_keys_to_action()
        # invert the keys to action mapping to lookup key combos by action
        action_to_keys = {v: k for k, v in old_keys_to_action.items()}
        # create a new mapping of keys to actions
        keys_to_action = {}
        # iterate over the actions and their byte values in this mapper
        for action, byte in self._action_map.items():
            # get the keys to press for the action
            keys = action_to_keys[byte]
            # set the keys value in the dictionary to the current discrete act
            keys_to_action[keys] = action

        return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]


class NesEnv(VideogameEnv):
    """Environment for working with the NES-py emulator."""

    @property
    def nes_env(self) -> "NESEnv":  # noqa: F821
        """Access the underlying NESEnv."""
        return self.gym_env.unwrapped

    def get_image(self) -> numpy.ndarray:
        """Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        return self.gym_env.screen.copy()

    def get_ram(self) -> numpy.ndarray:
        """Return a copy of the emulator environment."""
        return self.nes_env.ram.copy()

    def get_state(self, state: numpy.ndarray | None = None) -> numpy.ndarray:
        """Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        return self.gym_env.get_state(state)

    def set_state(self, state: numpy.ndarray) -> None:
        """Set the internal state of the simulation.

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
        try:
            super().close()
        except ValueError:  # pragma: no cover
            pass

    def __del__(self):
        """Tear down the environment."""
        try:
            self.close()
        except ValueError:  # pragma: no cover
            pass

    def render(self, mode="rgb_array"):  # noqa: ARG002
        """Render the environment."""
        return self.gym_env.screen.copy()


class MarioEnv(NesEnv):
    """Interface for using gym-super-mario-bros in plangym."""

    AVAILABLE_OBS_TYPES = {"coords", "rgb", "grayscale", "ram"}
    MOVEMENTS = {
        "complex": COMPLEX_MOVEMENT,
        "simple": SIMPLE_MOVEMENT,
        "right": RIGHT_ONLY,
    }

    def __init__(
        self,
        name: str,
        movement_type: str = "simple",
        original_reward: bool = False,
        **kwargs,
    ):
        """Initialize a MarioEnv.

        Args:
            name: Name of the environment.
            movement_type: One of {complex|simple|right}
            original_reward: If False return a custom reward based on mario position and level.
            **kwargs: passed to super().__init__.

        """
        self._movement_type = movement_type
        self._original_reward = original_reward
        super().__init__(name=name, **kwargs)

    def get_state(self, state: numpy.ndarray | None = None) -> numpy.ndarray:
        """Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        state = numpy.empty(250288, dtype=numpy.byte) if state is None else state
        state[-2:] = 0  # Some states use the last two bytes. Set to zero by default.
        return super().get_state(state)

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`NESEnv`` instance that the current class is wrapping."""
        from gym_super_mario_bros import make  # noqa: PLC0415
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT  # noqa: PLC0415

        env = make(self.name)
        gym_env = NESWrapper(JoypadSpace(env.unwrapped, COMPLEX_MOVEMENT))
        gym_env.reset()
        return gym_env

    def _update_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info["player_state"] = self.nes_env._player_state
        info["area"] = self.nes_env._area
        info["left_x_position"] = self.nes_env._left_x_position
        info["is_stage_over"] = self.nes_env._is_stage_over
        info["is_dying"] = self.nes_env._is_dying
        info["is_dead"] = self.nes_env._is_dead
        info["y_pixel"] = self.nes_env._y_pixel
        info["y_viewport"] = self.nes_env._y_viewport
        info["x_position_last"] = self.nes_env._x_position_last
        info["in_pipe"] = (info["player_state"] == 0x02) or (info["player_state"] == 0x03)  # noqa: PLR2004
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

    def get_coords_obs(
        self,
        obs: numpy.ndarray,
        info: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> numpy.ndarray:
        """Return the information contained in info as an observation if obs_type == "info"."""
        if self.obs_type == "coords":
            info = info or self._get_info()
            obs = numpy.array(
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

    def process_reward(self, reward, info, **kwargs) -> float:  # noqa: ARG002
        """Return a custom reward based on the x, y coordinates and level mario is in."""
        if not self._original_reward:
            world = int(info.get("world", 0))
            stage = int(info.get("stage", 0))
            x_pos = int(info.get("x_pos", 0))
            reward = (
                (world * 25000)
                + (stage * 5000)
                + x_pos
                + 10 * int(bool(info.get("in_pipe", 0)))
                + 100 * int(bool(info.get("flag_get", 0)))
                # + (abs(info["x_pos"] - info["x_position_last"]))
            )
        return reward

    def process_terminal(self, terminal, info, **kwargs) -> bool:  # noqa: ARG002
        """Return True if terminal or mario is dying."""
        return terminal or info.get("is_dying", False) or info.get("is_dead", False)

    def process_info(self, info, **kwargs) -> dict[str, Any]:  # noqa: ARG002
        """Add additional data to the info dictionary."""
        return self._update_info(info)
