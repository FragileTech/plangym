"""Plangym API implementation."""
from abc import ABC
from typing import Any, Dict, Iterable, Optional

import gym
import numpy

from plangym.core import PlangymEnv, wrap_callable


LIFE_KEY = "lifes"


class VideogameEnv(PlangymEnv, ABC):
    """Common interface for working with video games that run using an emulator."""

    AVAILABLE_OBS_TYPES = {"rgb", "grayscale", "ram"}
    DEFAULT_OBS_TYPE = "rgb"

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
        Initialize a :class:`VideogameEnv`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            frameskip: Number of times an action will be applied for each step
                in dt.
            episodic_life: Return ``end = True`` when losing a life.
            autoreset: Restart environment when reaching a terminal state.
            delay_setup: If ``True`` do not initialize the ``gym.Environment``
                and wait for ``setup`` to be called later.
            remove_time_limit: If True, remove the time limit from the environment.
            obs_type: One of {"rgb", "ram", "grayscale"}.
            mode: Integer or string indicating the game mode, when available.
            difficulty: Difficulty level of the game, when available.
            repeat_action_probability: Repeat the last action with this probability.
            full_action_space: Whether to use the full range of possible actions
                               or only those available in the game.
            render_mode: One of {None, "human", "rgb_aray"}.
            wrappers: Wrappers that will be applied to the underlying OpenAI env.
                      Every element of the iterable can be either a :class:`gym.Wrapper`
                      or a tuple containing ``(gym.Wrapper, kwargs)``.

        """
        self.episodic_life = episodic_life
        self._info_step = {LIFE_KEY: -1, "lost_life": False}
        super(VideogameEnv, self).__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            wrappers=wrappers,
            delay_setup=delay_setup,
            render_mode=render_mode,
            remove_time_limit=remove_time_limit,
            obs_type=obs_type,
            **kwargs,
        )

    @property
    def n_actions(self) -> int:
        """Return the number of actions available."""
        return self.action_space.n

    @staticmethod
    def get_lifes_from_info(info: Dict[str, Any]) -> int:
        """Return the number of lifes remaining in the current game."""
        return info.get("life", -1)

    def apply_action(self, action):
        """Evolve the environment for one time step applying the provided action."""
        obs, reward, terminal, info = super(VideogameEnv, self).apply_action(action=action)
        info[LIFE_KEY] = self.get_lifes_from_info(info)
        past_lifes = self._info_step.get(LIFE_KEY, -1)
        lost_life = past_lifes > info[LIFE_KEY] or self._info_step.get("lost_life")
        info["lost_life"] = lost_life
        terminal = (terminal or lost_life) if self.episodic_life else terminal
        return obs, reward, terminal, info

    def clone(self, **kwargs) -> "VideogameEnv":
        """Return a copy of the environment."""
        params = dict(
            episodic_life=self.episodic_life,
            obs_type=self.obs_type,
            render_mode=self.render_mode,
        )
        params.update(**kwargs)
        return super(VideogameEnv, self).clone(**params)

    def begin_step(self, action=None, dt=None, state=None, return_state: bool = None) -> None:
        """Perform setup of step variables before starting `step_with_dt`."""
        self._info_step = {LIFE_KEY: -1, "lost_life": False}
        super(VideogameEnv, self).begin_step(
            action=action,
            dt=dt,
            state=state,
            return_state=return_state,
        )

    def init_spaces(self) -> None:
        """Initialize the action_space and the observation_space of the environment."""
        super(VideogameEnv, self).init_spaces()
        if self.obs_type == "ram":
            if self.DEFAULT_OBS_TYPE == "ram":
                space = self.gym_env.observation_space
            else:
                ram_size = self.get_ram().shape
                space = gym.spaces.Box(low=0, high=255, dtype=numpy.uint8, shape=ram_size)
            self._obs_space = space

    def process_obs(self, obs, **kwargs):
        """Return the ram vector if obs_type == "ram" or and image otherwise."""
        obs = super(VideogameEnv, self).process_obs(obs, **kwargs)
        if self.obs_type == "ram" and self.DEFAULT_OBS_TYPE != "ram":
            obs = self.get_ram()
        return obs

    def get_ram(self) -> numpy.ndarray:
        """Return the ram of the emulator as a numpy array."""
        raise NotImplementedError()
