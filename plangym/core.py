"""Plangym API implementation."""
from abc import ABC
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import gym
from gym.spaces import Box, Space
from gym.wrappers.gray_scale_observation import GrayScaleObservation
import numpy

from plangym.utils import process_frame, remove_time_limit


wrap_callable = Union[Callable[[], gym.Wrapper], Tuple[Callable[..., gym.Wrapper], Dict[str, Any]]]


class PlanEnv(ABC):
    """
    Inherit from this class to adapt environments to different problems.

    Base class that establishes all needed methods and blueprints to work with
    Gym environments.
    """

    STATE_IS_ARRAY = True
    OBS_IS_ARRAY = True
    SINGLETON = False

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        delay_setup: bool = False,
        return_image: bool = False,
    ):
        """
        Initialize a :class:`Environment`.

        Args:
            name: Name of the environment.
            frameskip: Number of times ``step`` will be called with the same action.
            autoreset: Automatically reset the environment when the OpenAI environment
                returns ``end = True``.
            delay_setup: If ``True`` do not initialize the ``gym.Environment``
                and wait for ``setup`` to be called later (delayed setups are necessary
                when one requires to serialize the object environment or to have duplicated
                instances).
            return_image: If ``True`` add an "rgb" key in the `info` dictionary returned by `step`
             that contains an RGB representation of the environment state.

        """
        # Public attributes
        self._name = name
        self.frameskip = frameskip
        self.autoreset = autoreset
        self.delay_setup = delay_setup
        self._return_image = return_image
        # Attributes for tracking data during the step process
        self._n_step = 0
        self._obs_step = None
        self._reward_step = 0
        self._terminal_step = False
        self._info_step = {}
        self._action_step = None
        self._dt_step = None
        self._state_step = None
        self._return_state_step = None
        if not delay_setup:
            self.setup()

    def __del__(self):
        """Teardown the Environment when it is no longer needed."""
        return self.close()

    # Public API -----------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Return is the name of the environment."""
        return self._name

    @property
    def obs_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the observations returned by the Environment."""
        raise NotImplementedError()

    @property
    def action_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the actions applied to the Environment."""
        raise NotImplementedError()

    @property
    def unwrapped(self) -> "PlanEnv":
        """
        Completely unwrap this Environment.

        Returns:
            plangym.Environment: The base non-wrapped plangym.Environment instance

        """
        return self

    @property
    def return_image(self) -> bool:
        """
        Return `return_image` flag.

        If ``True`` add an "rgb" key in the `info` dictionary returned by `step` \
        that contains an RGB representation of the environment state.
        """
        return self._return_image

    def get_image(self) -> Union[None, numpy.ndarray]:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a grayscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        raise NotImplementedError()

    def step(
        self,
        action: Union[numpy.ndarray, int, float],
        state: numpy.ndarray = None,
        dt: int = 1,
        return_state: Optional[bool] = None,
    ) -> tuple:
        """
        Step the environment applying the supplied action.

        Optionally set the state to the supplied state before stepping it (the
        method prepares the environment in the given state, dismissing the current
        state, and applies the action afterwards).

        Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        In addition, the method allows the user to prepare the returned object,
        adding additional information and custom pre-processings via ``self.process_step``
        and ``self.get_step_tuple`` methods.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
            dt: Consecutive number of times that the action will be applied.
            return_state: Whether to return the state in the returned tuple. \
                If None, `step` will return the state if `state` was passed as a parameter.

        Returns:
            if state is None returns ``(observs, reward, terminal, info)``
            else returns ``(new_state, observs, reward, terminal, info)``

        """
        self.begin_step(action=action, state=state, dt=dt, return_state=return_state)
        if state is not None:
            self.set_state(state)
        obs, reward, terminal, info = self.step_with_dt(action=action, dt=dt)
        obs, reward, terminal, info = self.process_step(
            obs=obs,
            reward=reward,
            terminal=terminal,
            info=info,
        )
        step_data = self.get_step_tuple(
            obs=obs,
            reward=reward,
            terminal=terminal,
            info=info,
        )
        self.run_autoreset(step_data)  # Resets at the end to preserve the environment state.
        return step_data

    def reset(
        self,
        return_state: bool = True,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Restart the environment.

        Args:
            return_state: If ``True``, it will return the state of the environment.

        Returns:
            ``(state, obs)`` if ```return_state`` is ``True`` else return ``obs``.

        """
        obs = self.apply_reset()  # Returning info upon reset is not yet supported
        obs = self.process_obs(obs)
        return (self.get_state(), obs) if return_state else obs

    def step_batch(
        self,
        actions: Union[numpy.ndarray, Iterable[Union[numpy.ndarray, int]]],
        states: Union[numpy.ndarray, Iterable] = None,
        dt: Union[int, numpy.ndarray] = 1,
        return_state: bool = True,
    ) -> Tuple[Union[list, numpy.ndarray], ...]:
        """
        Allow stepping a vector of states and actions.

        Vectorized version of the `step` method. The signature and behaviour is
        the same as `step`, but taking a list of states, actions and dts as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            dt: int or array containing the consecutive that will be applied to each state.
                If array, the different values are distributed among the multiple environments
                (contrary to ``self.frameskip``, which is a common value for any instance).
            return_state: Whether to return the state in the returned tuple, depending on
                the boolean value. \
                If None, `step` will return the state if `state` was passed as a parameter.

        Returns:
            If return_state is `True`, the method returns `(new_states, observs, rewards, ends,
            infos)`. \
            If return_state is `False`, the method returns `(observs, rewards, ends, infos)`. \
            If return_state is `None`, the returned object depends on the states parameter.

        """
        dt_is_array = (isinstance(dt, numpy.ndarray) and dt.shape) or isinstance(dt, (list, tuple))
        dt = dt if dt_is_array else numpy.ones(len(actions), dtype=int) * dt
        no_states = states is None or states[0] is None
        states = [None] * len(actions) if no_states else states
        data = [
            self.step(action, state, dt=dt, return_state=return_state)
            for action, state, dt in zip(actions, states, dt)
        ]
        return tuple(list(x) for x in zip(*data))

    def clone(self, **kwargs) -> "PlanEnv":
        """Return a copy of the environment."""
        clone_kwargs = dict(
            name=self.name,
            frameskip=self.frameskip,
            autoreset=self.autoreset,
            delay_setup=self.delay_setup,
        )
        clone_kwargs.update(kwargs)
        return self.__class__(**clone_kwargs)

    def sample_action(self):  # pragma: no cover
        """
        Return a valid action that can be used to step the Environment.

        Implementing this method is optional, and it's only intended to make the
        testing process of the Environment easier.
        """
        pass

    # Internal API -----------------------------------------------------------------------------
    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1):
        """
        Take ``dt`` simulation steps and make the environment evolve in multiples\
        of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        The method performs any post-processing to the data after applying the action
        to the environment via ``self.process_apply_action``.

        This method neither computes nor returns any state.

        Args:
            action: Chosen action applied to the environment.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            Tuple containing ``(observs, reward, terminal, info)``.

        """
        self._n_step = 0
        for _ in range(int(dt) * self.frameskip):
            self._n_step += 1
            step_data = self.apply_action(action)  # Tuple (obs, reward, terminal, info)
            step_data = self.process_apply_action(*step_data)  # Post-processing to step_data
            self._obs_step, self._reward_step, self._terminal_step, self._info_step = step_data
            if self._terminal_step:
                break
        return step_data

    def run_autoreset(self, step_data):
        """Reset the environment automatically if needed."""
        *_, terminal, _ = step_data  # Assumes terminal, info are the last two elements
        if terminal and self.autoreset:
            self.reset(return_state=False)

    def get_step_tuple(
        self,
        obs,
        reward,
        terminal,
        info,
    ):
        """
        Prepare the tuple that step returns.

        This is a post processing state to have fine-grained control over what data \
        the current step is returning.

        By default it determines:
         - Return the state in the tuple (necessary information to save or load the game).
         - Adding the "rgb" key in the `info` dictionary containing an RGB \
         representation of the environment.

        Args:
            obs: Observation of the environment.
            reward: Reward signal.
            terminal: Boolean indicating if the environment is finished.
            info: Dictionary containing additional information about the environment.

        Returns:
            Tuple containing the environment data after calling `step`.
        """
        # Determine whether the method has to return the environment state
        default_mode = self._state_step is not None and self._return_state_step is None
        return_state = self._return_state_step or default_mode
        # Post processing
        obs = self.process_obs(
            obs=obs,
            reward=reward,
            terminal=terminal,
            info=info,
        )
        reward = self.process_reward(
            obs=obs,
            reward=reward,
            terminal=terminal,
            info=info,
        )
        terminal = self.process_terminal(
            obs=obs,
            reward=reward,
            terminal=terminal,
            info=info,
        )
        info = self.process_info(
            obs=obs,
            reward=reward,
            terminal=terminal,
            info=info,
        )
        step_data = (
            (self.get_state(), obs, reward, terminal, info)
            if return_state
            else (obs, reward, terminal, info)
        )
        return step_data

    def setup(self) -> None:
        """
        Run environment initialization.

        Including in this function all the code which makes the environment impossible
        to serialize will allow to dispatch the environment to different workers and
        initialize it once it's copied to the target process.
        """
        pass

    def begin_step(self, action=None, dt=None, state=None, return_state: bool = None):
        """Perform setup of step variables before starting `step_with_dt`."""
        self._n_step = 0
        self._obs_step = None
        self._reward_step = 0
        self._terminal_step = False
        self._info_step = {}
        self._action_step = action
        self._dt_step = dt
        self._state_step = state
        self._return_state_step = return_state

    def process_apply_action(
        self,
        obs,
        reward,
        terminal,
        info,
    ):
        """
        Perform any post-processing to the data returned by `apply_action`.

        Args:
            obs: Observation of the environment.
            reward: Reward signal.
            terminal: Boolean indicating if the environment is finished.
            info: Dictionary containing additional information about the environment.

        Returns:
            Tuple containing the processed data.
        """
        terminal = terminal or self._terminal_step
        reward = self._reward_step + reward
        info["n_step"] = int(self._n_step)
        return obs, reward, terminal, info

    def process_step(
        self,
        obs,
        reward,
        terminal,
        info,
    ):
        """
        Prepare the returned info dictionary.

        This is a post processing step to have fine-grained control over what data \
        the info dictionary contains.

        Args:
            obs: Observation of the environment.
            reward: Reward signal.
            terminal: Boolean indicating if the environment is finished.
            info: Dictionary containing additional information about the environment.

        Returns:
            Tuple containing the environment data after calling `step`.
        """
        info["n_step"] = info.get("n_step", int(self._n_step))
        info["dt"] = self._dt_step
        if self.return_image:
            info["rgb"] = self.get_image()
        return obs, reward, terminal, info

    def close(self) -> None:
        """Tear down the current environment."""
        pass

    # Developer API -----------------------------------------------------------------------------
    def process_obs(self, obs, **kwargs):
        """Perform optional computation for computing the observation returned by step."""
        return obs

    def process_reward(self, reward, **kwargs) -> float:
        """Perform optional computation for computing the reward returned by step."""
        return reward

    def process_terminal(self, terminal, **kwargs) -> bool:
        """Perform optional computation for computing the terminal flag returned by step."""
        return terminal

    def process_info(self, info, **kwargs) -> Dict[str, Any]:
        """Perform optional computation for computing the info dictionary returned by step."""
        return info

    def apply_action(self, action):
        """Evolve the environment for one time step applying the provided action."""
        raise NotImplementedError()

    def apply_reset(self, **kwargs):
        """Perform the resetting operation on the environment."""
        raise NotImplementedError()

    def get_state(self) -> Any:
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        raise NotImplementedError()

    def set_state(self, state: Any) -> None:
        """
        Set the internal state of the simulation. Overwrite current state by the given argument.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        raise NotImplementedError()


class PlangymEnv(PlanEnv):
    """Base class for implementing OpenAI ``gym`` environments in ``plangym``."""

    AVAILABLE_RENDER_MODES = {"human", "rgb_array", None}
    AVAILABLE_OBS_TYPES = {"coords", "rgb", "grayscale"}
    DEFAULT_OBS_TYPE = "coords"

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_setup: bool = False,
        remove_time_limit=True,
        render_mode: Optional[str] = None,
        episodic_life=False,
        obs_type=None,  # one of coords|rgb|grayscale|None
        return_image=False,
        **kwargs,
    ):
        """
        Initialize a :class:`PlangymEnv`.

        The user can read all private methods as instance properties.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            frameskip: Number of times an action will be applied for each ``dt``. Common
                argument to all environments.
            autoreset: Automatically reset the environment when the OpenAI environment
                returns ``end = True``.
            wrappers: Wrappers that will be applied to the underlying OpenAI env.
                Every element of the iterable can be either a :class:`gym.Wrapper`
                or a tuple containing ``(gym.Wrapper, kwargs)``.
            delay_setup: If ``True`` do not initialize the :class:`gym.Environment`
                and wait for ``setup`` to be called later.
            remove_time_limit: If True, remove the time limit from the environment.

        """
        self._render_mode = render_mode
        self._gym_env = None
        self._gym_env_kwargs = kwargs or {}  # Dictionary containing the gym.make arguments
        self._remove_time_limit = remove_time_limit
        self._wrappers = wrappers
        self._obs_space = None
        self._action_space = None
        if obs_type is not None:
            assert obs_type in self.AVAILABLE_OBS_TYPES, (
                f"obs_type {obs_type} is not accepted. Available "
                f"values are: {self.AVAILABLE_OBS_TYPES}"
            )
        self._obs_type = obs_type or self.DEFAULT_OBS_TYPE
        super(PlangymEnv, self).__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_setup=delay_setup,
            return_image=return_image,
        )

    def __str__(self):
        """Pretty print the environment."""
        text = (
            f"{self.__class__} {self.name} with parameters:\n"
            f"obs_type={self.obs_type}, render_mode={self.render_mode}\n"
            f"frameskip={self.frameskip}, obs_shape={self.obs_shape},\n"
            f"action_shape={self.action_shape}"
        )
        return text

    def __repr__(self):
        """Pretty print the environment."""
        return str(self)

    @property
    def gym_env(self):
        """Return the instance of the environment that is being wrapped by plangym."""
        if self._gym_env is None and not self.SINGLETON:
            self.setup()
        return self._gym_env

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the *observations* returned by the Environment."""
        return self.observation_space.shape

    @property
    def obs_type(self) -> str:
        """Return the *type* of observation returned by the environment."""
        return self._obs_type

    @property
    def observation_space(self) -> Space:
        """Return the *observation_space* of the environment."""
        return self._obs_space

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the *actions* applied to the Environment."""
        return self.action_space.shape

    @property
    def action_space(self) -> Space:
        """Return the *action_space* of the environment."""
        return self._action_space

    @property
    def reward_range(self):
        """Return the *reward_range* of the environment."""
        if hasattr(self.gym_env, "reward_range"):
            return self.gym_env.reward_range

    @property
    def metadata(self):
        """Return the *metadata* of the environment."""
        if hasattr(self.gym_env, "metadata"):
            return self.gym_env.metadata
        return {"render_modes": [None, "human", "rgb_array"]}

    @property
    def render_mode(self) -> Union[None, str]:
        """Return how the game will be rendered. Values: None | human | rgb_array."""
        return self._render_mode

    @property
    def remove_time_limit(self) -> bool:
        """Return True if the Environment can only be stepped for a limited number of times."""
        return self._remove_time_limit

    def setup(self):
        """
        Initialize the target :class:`gym.Env` instance.

        The method calls ``self.init_gym_env`` to initialize the :class:``gym.Env`` instance.
        It removes time limits if needed and applies wrappers introduced by the user.
        """
        self._gym_env = self.init_gym_env()
        if self.remove_time_limit:
            self._gym_env = remove_time_limit(self._gym_env)
        if self._wrappers is not None:
            self.apply_wrappers(self._wrappers)
        self.init_spaces()

    def init_spaces(self):
        """Initialize the action_space and observation_space of the environment."""
        self._init_action_space()
        if self.obs_type == "rgb":
            self._init_obs_space_rgb()
        elif self.obs_type == "grayscale":
            self._init_obs_space_grayscale()
        elif self.obs_type == "coords":
            self._init_obs_space_coords()
        if self.observation_space is None:
            self._obs_space = self.gym_env.observation_space

    def _init_action_space(self):
        self._action_space = self.gym_env.action_space

    def _init_obs_space_rgb(self):
        if self.DEFAULT_OBS_TYPE == "rgb":
            self._obs_space = self.gym_env.observation_space
        else:
            img_shape = self.get_image().shape
            self._obs_space = Box(low=0, high=255, dtype=numpy.uint8, shape=img_shape)

    def _init_obs_space_grayscale(self):

        if self.DEFAULT_OBS_TYPE == "grayscale":
            self._obs_space = self.gym_env.observation_space
        elif self.DEFAULT_OBS_TYPE == "rgb":
            self._obs_space = self.gym_env.observation_space
            self._gym_env = GrayScaleObservation(self._gym_env)
            self._obs_space = self._gym_env.observation_space
        else:
            shape = self.get_image().shape
            self._obs_space = Box(low=0, high=255, dtype=numpy.uint8, shape=(shape[0], shape[1]))

    def _init_obs_space_coords(self):
        if self.DEFAULT_OBS_TYPE == "coords":
            if hasattr(self.gym_env, "observation_space"):
                self._obs_space = self.gym_env.observation_space
            else:
                raise NotImplementedError("No observation_space implemented.")
        else:
            img = self.reset(return_state=False)
            cords = self.get_coords_obs(img)
            self._obs_space = Box(
                low=-numpy.inf,
                high=numpy.inf,
                dtype=numpy.float32,
                shape=cords.shape,
            )

    def get_image(self) -> numpy.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB).
        """
        if hasattr(self.gym_env, "render"):
            return self.gym_env.render(mode="rgb_array")
        raise NotImplementedError()

    def apply_reset(
        self,
        return_state: bool = True,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Restart the environment.

        Args:
            return_state: If ``True`` it will return the state of the environment.

        Returns:
            ``(state, obs)`` if ```return_state`` is ``True`` else return ``obs``.

        """
        if self.gym_env is None and self.delay_setup:
            self.setup()
        return self.gym_env.reset()

    def apply_action(self, action):
        """
        Evolve the environment for one time step applying the provided action.

        Accumulate rewards and calculate terminal flag after stepping the environment.
        """
        obs, reward, terminal, info = self.gym_env.step(action)
        return obs, reward, terminal, info

    def sample_action(self) -> Union[int, numpy.ndarray]:
        """Return a valid action that can be used to step the environment chosen at random."""
        if hasattr(self.action_space, "sample"):
            return self.action_space.sample()
        return self.gym_env.action_space.sample()  # pragma: no cover

    def clone(self, **kwargs) -> "PlangymEnv":
        """Return a copy of the environment."""
        env_kwargs = dict(
            wrappers=self._wrappers,
            remove_time_limit=self._remove_time_limit,
            render_mode=self.render_mode,
        )
        env_kwargs.update(kwargs)
        env_kwargs.update(self._gym_env_kwargs)
        env: PlangymEnv = super(PlangymEnv, self).clone(**env_kwargs)
        return env

    def close(self):
        """Close the underlying :class:`gym.Env`."""
        if hasattr(self, "_gym_env") and hasattr(self._gym_env, "close"):
            return self._gym_env.close()
        self._gym_env = None

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:``gym.Env`` instance that the current class is wrapping."""
        gym_env: gym.Env = gym.make(self.name, **self._gym_env_kwargs)
        gym_env.reset()
        return gym_env

    def seed(self, seed=None):
        """Seed the underlying :class:`gym.Env`."""
        if hasattr(self.gym_env, "seed"):
            return self.gym_env.seed(seed)

    def apply_wrappers(self, wrappers: Iterable[wrap_callable]):
        """Wrap the underlying OpenAI gym environment."""
        for item in wrappers:
            if isinstance(item, tuple):
                wrapper, kwargs = item
                if isinstance(kwargs, dict):
                    self.wrap(wrapper, **kwargs)
                elif isinstance(kwargs, (list, tuple)):
                    self.wrap(wrapper, *kwargs)
                else:
                    self.wrap(wrapper, kwargs)
            else:
                self.wrap(item)

    def wrap(self, wrapper: Callable, *args, **kwargs):
        """Apply a single OpenAI gym wrapper to the environment."""
        self._gym_env = wrapper(self.gym_env, *args, **kwargs)

    def render(self, mode="human"):
        """Render the environment using OpenGL. This wraps the OpenAI render method."""
        if hasattr(self.gym_env, "render"):
            return self.gym_env.render(mode=mode)
        raise NotImplementedError()

    def process_obs(self, obs, **kwargs):
        """
        Perform optional computation for computing the observation returned by step.

        This is a post processing step to have fine-grained control over the returned
        observation.
        """
        if self.obs_type == "coords":
            return self.get_coords_obs(obs, **kwargs)
        elif self.obs_type == "rgb":
            return self.get_rgb_obs(obs, **kwargs)
        elif self.obs_type == "grayscale":
            return self.get_grayscale_obs(obs, **kwargs)
        return obs

    def get_coords_obs(self, obs, **kwargs):
        """Calculate the observation returned by `step` when obs_type == "coords"."""
        if self.DEFAULT_OBS_TYPE == "coords":
            return obs
        raise NotImplementedError()

    def get_rgb_obs(self, obs, **kwargs):
        """Calculate the observation returned by `step` when obs_type == "rgb"."""
        if self.DEFAULT_OBS_TYPE == "rgb":
            return obs
        return self.get_image()

    def get_grayscale_obs(self, obs, **kwargs):
        """Calculate the observation returned by `step` when obs_type == "grayscale"."""
        obs = self.get_rgb_obs(obs, **kwargs)
        return process_frame(obs, mode="L")
