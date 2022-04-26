"""Implement the ``plangym`` API for retro environments."""
from typing import Any, Dict, Iterable, Optional

import gym
from gym import spaces
import numpy

from plangym.core import wrap_callable
from plangym.videogames.env import VideogameEnv


class ActionDiscretizer(gym.ActionWrapper):
    """Wrap a gym-retro environment and make it use discrete actions for the Sonic game."""

    def __init__(self, env):
        """Initialize a :class`ActionDiscretizer`."""
        super(ActionDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [
            ["LEFT"],
            ["RIGHT"],
            ["LEFT", "DOWN"],
            ["RIGHT", "DOWN"],
            ["DOWN"],
            ["DOWN", "B"],
            ["B"],
        ]
        self._actions = []
        for action in actions:
            arr = numpy.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = spaces.Discrete(len(self._actions))

    def action(self, a) -> int:  # pylint: disable=W0221
        """Return the corresponding action in the emulator's format."""
        return self._actions[a].copy()


class RetroEnv(VideogameEnv):
    """Environment for playing ``gym-retro`` games."""

    AVAILABLE_OBS_TYPES = {"coords", "rgb", "grayscale", "ram"}
    SINGLETON = True

    def __init__(
        self,
        name: str,
        frameskip: int = 5,
        episodic_life: bool = False,
        autoreset: bool = True,
        delay_setup: bool = False,
        remove_time_limit: bool = True,
        obs_type: str = "rgb",  # ram | rgb | grayscale
        render_mode: Optional[str] = None,  # None | human | rgb_array
        wrappers: Iterable[wrap_callable] = None,
        **kwargs,
    ):
        """
        Initialize a :class:`RetroEnv`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            frameskip: Number of times an action will be applied for each step \
                in dt.
            episodic_life: Return ``end = True`` when losing a life.
            autoreset: Restart environment when reaching a terminal state.
            delay_setup: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``setup`` to be called later.
            remove_time_limit: If True, remove the time limit from the environment.
            obs_type: One of {"rgb", "ram", "grayscale"}.
            render_mode: One of {None, "human", "rgb_aray"}.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
        """
        super(RetroEnv, self).__init__(
            name=name,
            frameskip=frameskip,
            episodic_life=episodic_life,
            autoreset=autoreset,
            delay_setup=delay_setup,
            remove_time_limit=remove_time_limit,
            obs_type=obs_type,  # ram | rgb | grayscale
            render_mode=render_mode,  # None | human | rgb_array
            wrappers=wrappers,
            **kwargs,
        )

    def __getattr__(self, item):
        """Forward getattr to self.gym_env."""
        return getattr(self.gym_env, item)

    @staticmethod
    def get_win_condition(info: Dict[str, Any]) -> bool:  # pragma: no cover
        """Get win condition for games that have the end of the screen available."""
        end_screen = info.get("screen_x", 0) >= info.get("screen_x_end", 1e6)
        terminal = info.get("x", 0) >= info.get("screen_x_end", 1e6) or end_screen
        return terminal

    def get_ram(self) -> numpy.ndarray:
        """Return the ram of the emulator as a numpy array."""
        return self.get_state()  # .copy()

    def clone(self, **kwargs) -> "RetroEnv":
        """Return a copy of the environment with its initialization delayed."""
        default_kwargs = dict(
            name=self.name,
            frameskip=self.frameskip,
            wrappers=self._wrappers,
            episodic_life=self.episodic_life,
            autoreset=self.autoreset,
            delay_setup=self.delay_setup,
            obs_type=self.obs_type,
        )
        default_kwargs.update(kwargs)
        return super(RetroEnv, self).clone(**default_kwargs)

    def init_gym_env(self) -> gym.Env:
        """Initialize the retro environment."""
        import retro

        if self._gym_env is not None:
            self._gym_env.close()
        gym_env = retro.make(self.name, **self._gym_env_kwargs)
        return gym_env

    def get_state(self) -> numpy.ndarray:
        """Get the state of the retro environment."""
        state = self.gym_env.em.get_state()
        return numpy.frombuffer(state, dtype=numpy.uint8)

    def set_state(self, state: numpy.ndarray):
        """Set the state of the retro environment."""
        raw_state = state.tobytes()
        self.gym_env.em.set_state(raw_state)
        return state

    def close(self):
        """Close the underlying :class:`gym.Env`."""
        if hasattr(self, "_gym_env") and hasattr(self._gym_env, "close"):
            import gc

            self._gym_env.close()
            gc.collect()
