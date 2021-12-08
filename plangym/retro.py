"""Implement the ``plangym`` API for retro environments."""
from typing import Any, Dict, Iterable, Tuple, Union

import numpy
from PIL import Image

from plangym.core import PlanEnvironment, wrap_callable


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


class RetroEnvironment(PlanEnvironment):
    """Environment for playing ``gym-retro`` games."""

    SINGLETON = True

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        episodic_live: bool = False,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_init: bool = False,
        obs_type: str = "rgb",  # ram | rgb | grayscale
        height: int = None,  # 100,
        width: int = None,  # 100,
        **kwargs,
    ):
        """
        Initialize a :class:`PlanEnvironment`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            frameskip: Number of times an action will be applied for each ``dt``.
            episodic_live: Return ``end = True`` when losing a live.
            autoreset: Automatically reset the environment when the OpenAI environment
                      returns ``end = True``.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``init_env`` to be called later.
            height: Resize the observation to have this height.
            width: Resize the observations to have this width.
            **kwargs: Passed to ``retro.make``.
        """
        self._wrappers = wrappers
        self.episodic_life = episodic_live
        self._gym_env = None
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.metadata = None
        self.gym_env_kwargs = kwargs
        self.height = height
        self.width = width
        self.obs_type = obs_type
        super(PlanEnvironment, self).__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_init=delay_init,
        )

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return self.observation_space.shape if self.gym_env is not None else ()

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return self.action_space.shape if self.gym_env is not None else ()

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
        env = retro.make(self.name, **self.gym_env_kwargs).unwrapped
        if self._wrappers is not None:
            self.apply_wrappers(self._wrappers)
        self._gym_env = env
        self.action_space = self.gym_env.action_space
        self.observation_space = (
            self.gym_env.observation_space
            if self.observation_space is None
            else self.observation_space
        )
        self.action_space = (
            self.gym_env.action_space if self.action_space is None else self.action_space
        )
        self.reward_range = (
            self.gym_env.reward_range if self.reward_range is None else self.reward_range
        )
        self.metadata = self.gym_env.metadata if self.metadata is None else self.metadata

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
            observ = self.get_state().copy() if ram_obs else self.process_obs(observ)
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
