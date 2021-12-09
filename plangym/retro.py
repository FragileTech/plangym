"""Implement the ``plangym`` API for retro environments."""
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from gym import spaces
import numpy
from PIL import Image

from plangym.core import VideogameEnvironment, wrap_callable


try:
    import retro

except ModuleNotFoundError:
    print("Please install OpenAI retro")


def resize_frame(
    frame: numpy.ndarray,
    width: int,
    height: int,
    mode: str = "RGB",
) -> numpy.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize(size=(width, height))
    return numpy.array(frame)


class RetroEnvironment(VideogameEnvironment):
    """Environment for playing ``gym-retro`` games."""

    SINGLETON = True

    def __init__(
        self,
        name: str,
        frameskip: int = 5,
        episodic_live: bool = False,
        autoreset: bool = True,
        delay_init: bool = False,
        remove_time_limit: bool = True,
        obs_type: str = "rgb",  # ram | rgb | grayscale
        mode: int = 0,  # game mode, see Machado et al. 2018
        difficulty: int = 0,  # game difficulty, see Machado et al. 2018
        repeat_action_probability: float = 0.0,  # Sticky action probability
        full_action_space: bool = False,  # Use all actions
        render_mode: Optional[str] = None,  # None | human | rgb_array
        possible_to_win: bool = False,
        wrappers: Iterable[wrap_callable] = None,
        array_state: bool = True,
        height: int = None,  # 100,
        width: int = None,  # 100,
        **kwargs,
    ):
        """
        Initialize a :class:`RetroEnvironment`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            frameskip: Number of times an action will be applied for each step \
                in dt.
            episodic_live: Return ``end = True`` when losing a life.
            autoreset: Restart environment when reaching a terminal state.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``init_env`` to be called later.
            remove_time_limit: If True, remove the time limit from the environment.
            obs_type: One of {"rgb", "ram", "gryscale"}.
            mode: Alias for state. Passed to retro.make().
            difficulty: Difficulty level of the game, when available.
            repeat_action_probability: Repeat the last action with this probability.
            full_action_space: Wheter to use the full range of possible actions \
                              or only those available in the game.
            render_mode: One of {None, "human", "rgb_aray"}.
            possible_to_win: It is possible to finish the Atari game without \
                            getting a terminal state that is not out of bounds \
                            or doest not involve losing a life.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            array_state: Whether to return the state of the environment as a numpy array.
        """
        kwargs["state"] = kwargs.get("state", mode)
        self.gym_env_kwargs = kwargs
        self.height = height
        self.width = width
        self._obs_space = None
        super(RetroEnvironment, self).__init__(
            name=name,
            frameskip=frameskip,
            episodic_live=episodic_live,
            autoreset=autoreset,
            delay_init=delay_init,
            remove_time_limit=remove_time_limit,
            obs_type=obs_type,  # ram | rgb | grayscale
            mode=mode,  # game mode, see Machado et al. 2018
            difficulty=difficulty,  # game difficulty, see Machado et al. 2018
            repeat_action_probability=repeat_action_probability,  # Sticky action probability
            full_action_space=full_action_space,  # Use all actions
            render_mode=render_mode,  # None | human | rgb_array
            possible_to_win=possible_to_win,
            wrappers=wrappers,
        )

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return self.observation_space.shape if self.gym_env is not None else ()

    @property
    def observation_space(self):
        """Return the observation_space of the environment."""
        return self._obs_space

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return self.action_space.shape if self.gym_env is not None else ()

    def get_ram(self) -> numpy.ndarray:
        """Return the ram of the emulator as a numpy array."""
        return self.get_state().copy()

    def clone(self) -> "RetroEnvironment":
        """Return a copy of the environment with its initialization delayed."""
        return RetroEnvironment(
            name=self.name,
            frameskip=self.frameskip,
            wrappers=self._wrappers,
            episodic_live=self.episodic_life,
            autoreset=self.autoreset,
            delay_init=self.delay_init,
            obs_type=self.obs_type,
            height=self.height,
            width=self.width,
        )

    def init_env(self):
        """Initialize the internal retro environment and its class attributes."""
        if self._gym_env is not None:
            self._gym_env.close()
        env = retro.make(self.name, **self.gym_env_kwargs).unwrapped
        if self._wrappers is not None:
            self.apply_wrappers(self._wrappers)
        self._gym_env = env
        if self.obs_type == "ram":
            ram_size = self.get_ram().shape
            self._obs_space = spaces.Box(low=0, high=255, dtype=numpy.uint8, shape=ram_size)
        self._obs_space = self._obs_space or self.gym_env.observation_space

    def __getattr__(self, item):
        """Forward getattr to self.gym_env."""
        return getattr(self.gym_env, item)

    def get_state(self) -> numpy.ndarray:
        """Get the state of the retro environment."""
        state = self.gym_env.em.get_state()
        return numpy.frombuffer(state, dtype=numpy.int32)

    def set_state(self, state: numpy.ndarray):
        """Set the state of the retro environment."""
        raw_state = state.tobytes()
        self.gym_env.em.set_state(raw_state)
        return state

    def step(
        self,
        action: Union[numpy.ndarray, int],
        state: numpy.ndarray = None,
        dt: int = 1,
    ) -> tuple:
        """
        Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.frameskip``.

        The info dictionary will contain a boolean called '`lost_live'` that will
        be ``True`` if a life was lost during the current step.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            if states is None returns ``(observs, rewards, ends, infos)``
            else returns ``(new_states, observs, rewards, ends, infos)``

        """
        data = super(RetroEnvironment, self).step(action=action, state=state, dt=dt)
        ram_obs = self.obs_type == "ram"
        if state is None:
            observ, reward, terminal, info = data
            observ = self.get_ram() if ram_obs else self.process_obs(observ)
            return observ, reward, terminal, info
        else:
            state, observ, reward, terminal, info = data
            observ = state.copy() if ram_obs else self.process_obs(observ)
            return state, observ, reward, terminal, info

    def process_obs(self, obs):
        """Resize the observations to the target size and transform them to grayscale."""
        obs = (
            resize_frame(obs, self.height, self.width)
            if self.width is not None and self.height is not None
            else obs
        )
        return obs

    @staticmethod
    def get_win_condition(info: Dict[str, Any]) -> bool:
        """Get win condition for games that have the end of the screen available."""
        end_screen = info.get("screen_x", 0) >= info.get("screen_x_end", 1e6)
        terminal = info.get("x", 0) >= info.get("screen_x_end", 1e6) or end_screen
        return terminal

    def reset(self, return_state: bool = True):
        """
        Reset the environment and return the first ``observation``, or the first \
        ``(state, obs)`` tuple.

        Args:
            return_state: If ``True`` return a also the initial state of the env.

        Returns:
            ``Observation`` of the environment if `return_state` is ``False``. \
            Otherwise return ``(state, obs)`` after reset.

        """
        if self.gym_env is None and self.delay_init:
            self.init_env()
        obs = self.gym_env.reset()
        if self.obs_type == "ram":
            obs = self.get_state().copy()
        else:
            obs = (
                resize_frame(obs, self.height, self.width)
                if self.width is not None and self.height is not None
                else obs
            )
        if not return_state:
            return obs
        else:
            return self.get_state(), obs
