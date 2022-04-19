"""Plangym API implementation."""
from abc import ABC
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple, Union

import gym
from gym.envs.registration import registry as gym_registry
from gym.spaces import Space
import numpy
import numpy as np

from plangym.utils import remove_time_limit, remove_time_limit_from_spec


wrap_callable = Union[Callable[[], gym.Wrapper], Tuple[Callable[..., gym.Wrapper], Dict[str, Any]]]

LIFE_KEY = "lifes"


class BaseEnvironment(ABC):
    """Inherit from this class to adapt environments to different problems."""

    STATE_IS_ARRAY = True
    RETURNS_GYM_TUPLE = True
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
                and wait for ``setup`` to be called later.
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

    @property
    def unwrapped(self) -> "BaseEnvironment":
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

    def get_image(self) -> Union[None, np.ndarray]:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        return None

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

    def step(
        self,
        action: Union[numpy.ndarray, int, float],
        state: numpy.ndarray = None,
        dt: int = 1,
        return_state: Optional[bool] = None,
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

    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1):
        """
        Take ``dt`` simulation steps and make the environment evolve in multiples\
        of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        Args:
            action: Chosen action applied to the environment.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            Tuple containing ``(observs, reward, terminal, info)``.

        """
        self._n_step = 0
        for _ in range(int(dt)):
            for _ in range(self.frameskip):
                step_data = self.apply_action(action)  # (obs, reward, terminal, info)
                step_data = self.process_apply_action(*step_data)
                self._obs_step, self._reward_step, self._terminal_step, self._info_step = step_data
                self._n_step += 1
                if self._terminal_step:
                    break
            if self._terminal_step:
                break
        return step_data

    def run_autoreset(self, step_data):
        """Reset the environment automatically if needed."""
        *_, terminal, _ = step_data  # Assumes terminal, info are the last two elements
        if terminal and self.autoreset:
            self.reset(return_state=False)

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
        info["n_step"] = int(self._n_step)
        info["dt"] = self._dt_step
        if self.return_image:
            info["rgb"] = self.get_image()
        return obs, reward, terminal, info

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
        that step is returning.

        By default it determines:
         - Return the state in the tuple.
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
        default_mode = self._state_step is not None and self._return_state_step is None
        return_state = self._return_state_step or default_mode
        step_data = (
            (self.get_state(), obs, reward, terminal, info)
            if return_state
            else (obs, reward, terminal, info)
        )
        return step_data

    def step_batch(
        self,
        actions: Union[numpy.ndarray, Iterable[Union[numpy.ndarray, int]]],
        states: Union[numpy.ndarray, Iterable] = None,
        dt: Union[int, numpy.ndarray] = 1,
    ) -> Tuple[Union[list, numpy.ndarray], ...]:
        """
        Vectorized version of the `step` method. It allows to step a vector of \
        states and actions.

        The signature and behaviour is the same as `step`, but taking a list of
        states, actions and dts as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            dt: int or array containing the consecutive that will be applied to each state.

        Returns:
            if states is `None` returns `(observs, rewards, ends, infos)`
            else returns `(new_states, observs, rewards, ends, infos)`.

        """
        dt_is_array = (isinstance(dt, numpy.ndarray) and dt.shape) or isinstance(dt, (list, tuple))
        dt = dt if dt_is_array else numpy.ones(len(actions), dtype=int) * dt
        no_states = states is None or states[0] is None
        states = [None] * len(actions) if no_states else states
        data = [self.step(action, state, dt=dt) for action, state, dt in zip(actions, states, dt)]
        return tuple(list(x) for x in zip(*data))

    def clone(self, **kwargs) -> "BaseEnvironment":
        """Return a copy of the environment."""
        clone_kwargs = dict(
            name=self.name,
            frameskip=self.frameskip,
            autoreset=self.autoreset,
            delay_setup=self.delay_setup,
        )
        clone_kwargs.update(kwargs)
        return self.__class__(**clone_kwargs)

    def setup(self) -> None:
        """
        Run environment initialization.

        Including in this function all the code which makes the environment impossible
        to serialize will allow to dispatch the environment to different workers and
        initialize it once it's copied to the target process.
        """
        pass

    def close(self) -> None:
        """Tear down the current environment."""
        pass

    def sample_action(self):
        """
        Return a valid action that can be used to step the Environment.

        Implementing this method is optional, and it's only intended to make the
        testing process of the Environment easier.
        """
        pass

    def apply_action(self, action):
        """Evolve the environment for one time step applying the provided action."""
        raise NotImplementedError()

    def reset(
        self,
        return_state: bool = True,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Restart the environment.

        Args:
            return_state: If ``True`` it will return the state of the environment.

        Returns:
            ``obs`` if ```return_state`` is ``True`` else return ``(state, obs)``.

        """
        raise NotImplementedError()

    def get_state(self) -> Any:
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        raise NotImplementedError()

    def set_state(self, state: Any) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        raise NotImplementedError()


class PlanEnvironment(BaseEnvironment):
    """Base class for implementing OpenAI ``gym`` environments in ``plangym``."""

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
    ):
        """
        Initialize a :class:`PlanEnvironment`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            frameskip: Number of times an action will be applied for each ``dt``.
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
        self.episodic_life = episodic_life
        self._remove_time_limit = remove_time_limit
        self._wrappers = wrappers
        super(PlanEnvironment, self).__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_setup=delay_setup,
            return_image=render_mode == "rgb_array",
        )

    @property
    def gym_env(self):
        """Return the instance of the environment that is being wrapped by plangym."""
        if self._gym_env is None and not self.SINGLETON:
            self.setup()
        return self._gym_env

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return self.observation_space.shape

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return self.action_space.shape

    @property
    def action_space(self) -> Space:
        """Return the action_space of the environment."""
        return self.gym_env.action_space

    @property
    def observation_space(self) -> Space:
        """Return the observation_space of the environment."""
        return self.gym_env.observation_space

    @property
    def reward_range(self):
        """Return the reward_range of the environment."""
        if hasattr(self.gym_env, "reward_range"):
            return self.gym_env.reward_range

    @property
    def metadata(self):
        """Return the metadata of the environment."""
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
        """Initialize the target :class:`gym.Env` instance."""
        self._gym_env = self.init_gym_env()
        if self._wrappers is not None:
            self.apply_wrappers(self._wrappers)

    def get_image(self) -> np.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        if hasattr(self.gym_env, "render"):
            return self.gym_env.render(mode="rgb_array")

    def reset(
        self,
        return_state: bool = True,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Restart the environment.

        Args:
            return_state: If ``True`` it will return the state of the environment.

        Returns:
            ``obs`` if ```return_state`` is ``True`` else return ``(state, obs)``.

        """
        if self.gym_env is None and self.delay_setup:
            self.setup()
        obs = self.gym_env.reset()
        return (self.get_state(), obs) if return_state else obs

    def apply_action(self, action):
        """Accumulate rewards and calculate terminal flag after stepping the environment."""
        obs, reward, terminal, info = self.gym_env.step(action)
        return obs, reward, terminal, info

    def sample_action(self) -> Union[int, np.ndarray]:
        """Return a valid action that can be used to step the Environment chosen at random."""
        if hasattr(self.action_space, "sample"):
            return self.action_space.sample()

    def clone(self, **kwargs) -> "PlanEnvironment":
        """Return a copy of the environment."""
        env_kwargs = dict(
            wrappers=self._wrappers,
            remove_time_limit=self._remove_time_limit,
            render_mode=self.render_mode,
        )
        env_kwargs.update(kwargs)
        env: PlanEnvironment = super(PlanEnvironment, self).clone(**env_kwargs)
        return env

    def close(self):
        """Close the underlying :class:`gym.Env`."""
        if hasattr(self, "_gym_env") and hasattr(self._gym_env, "close"):
            return self._gym_env.close()

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`gym.Env`` instance that the current class is wrapping."""
        # Remove any undocumented wrappers
        spec = gym_registry.spec(self.name)
        if self.remove_time_limit:
            remove_time_limit_from_spec(spec)
        gym_env: gym.Env = spec.make()
        if self.remove_time_limit:
            gym_env = remove_time_limit(gym_env)
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
                self.wrap(wrapper, **kwargs)
            else:
                self.wrap(item)

    def wrap(self, wrapper: Callable, *args, **kwargs):
        """Apply a single OpenAI gym wrapper to the environment."""
        self._gym_env = wrapper(self.gym_env, *args, **kwargs)

    @staticmethod
    def get_lifes_from_info(info: Dict[str, Any]) -> int:
        """Return the number of lifes remaining in the current game."""
        return info.get(LIFE_KEY, -1)

    @staticmethod
    def get_win_condition(info: Dict[str, Any]) -> bool:
        """Return ``True`` if the current state corresponds to winning the game."""
        return False

    def terminal_condition(self, old_info, new_info, terminal, *args, **kwargs) -> bool:
        """Calculate a new terminal condition using the info data."""
        return False

    def render(self, mode="human"):
        """Render the environment using OpenGL. This wraps the OpenAI render method."""
        if hasattr(self.gym_env, "render"):
            return self.gym_env.render(mode=mode)


class VideogameEnvironment(PlanEnvironment):
    """Common interface for working with video games that run using an emulator."""

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
    ):
        """
        Initialize a :class:`VideogameEnvironment`.

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
        self._obs_type = obs_type
        self.episodic_life = episodic_life
        self._info_step = {LIFE_KEY: -1, "lost_life": False}
        super(VideogameEnvironment, self).__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            wrappers=wrappers,
            delay_setup=delay_setup,
            render_mode=render_mode,
            remove_time_limit=remove_time_limit,
        )

    @property
    def obs_type(self) -> str:
        """Return the type of observation returned by the environment."""
        return self._obs_type

    @property
    def n_actions(self) -> int:
        """Return the number of actions available."""
        return self.gym_env.action_space.n

    def apply_action(self, action):
        """Evolve the environment for one time step applying the provided action."""
        obs, reward, terminal, info = super(VideogameEnvironment, self).apply_action(action=action)
        info[LIFE_KEY] = self.get_lifes_from_info(info)
        past_lifes = self._info_step.get(LIFE_KEY, -1)
        lost_life = past_lifes > info[LIFE_KEY] or self._info_step.get("lost_life")
        info["lost_life"] = lost_life
        terminal = (terminal or lost_life) if self.episodic_life else terminal
        return obs, reward, terminal, info

    def clone(self, **kwargs) -> "VideogameEnvironment":
        """Return a copy of the environment."""
        params = dict(
            episodic_life=self.episodic_life,
            obs_type=self.obs_type,
            render_mode=self.render_mode,
        )
        params.update(**kwargs)
        return super(VideogameEnvironment, self).clone(**params)

    def begin_step(self, action=None, dt=None, state=None, return_state: bool = None):
        """Perform setup of step variables before starting `step_with_dt`."""
        self._info_step = {LIFE_KEY: -1, "lost_life": False}
        super(VideogameEnvironment, self).begin_step(
            action=action,
            dt=dt,
            state=state,
            return_state=return_state,
        )

    def get_ram(self) -> np.ndarray:
        """Return the ram of the emulator as a numpy array."""
        raise NotImplementedError()


class VectorizedEnvironment(PlanEnvironment, ABC):
    """
    Base class that defines the API for working with vectorized environments.

    A vectorized environment allows to step several copies of the environment in parallel
    when calling ``step_batch``.

    It creates a local copy of the environment that is the target of all the other
    methods of :class:`BaseEnvironment`. In practise, a :class:`VectorizedEnvironment`
    acts as a wrapper of an environment initialized with the provided parameters when calling
    __init__.

    """

    def __init__(
        self,
        env_class,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        delay_setup: bool = False,
        n_workers: int = 8,
        **kwargs,
    ):
        """
        Initialize a :class:`VectorizedEnvironment`.

        Args:
            env_class: Class of the environment to be wrapped.
            name: Name of the environment.
            frameskip: Number of times ``step`` will be called with the same action.
            autoreset: Ignored. Always set to True. Automatically reset the environment
                when the OpenAI environment returns ``end = True``.
            delay_setup: If ``True`` do not initialize the :class:`gym.Environment`
                and wait for ``setup`` to be called later.
            env_callable: Callable that returns an instance of the environment
                that will be parallelized.
            n_workers:  Number of workers that will be used to step the env.
            **kwargs: Additional keyword arguments passed to env_class.__init__.

        """
        self._n_workers = n_workers
        self._env_class = env_class
        self._env_kwargs = kwargs
        self._plangym_env = None
        self.SINGLETON = env_class.SINGLETON if hasattr(env_class, "SINGLETON") else False
        self.RETURNS_GYM_TUPLE = (
            env_class.RETURNS_GYM_TUPLE if hasattr(env_class, "RETURNS_GYM_TUPLE") else True
        )
        self.STATE_IS_ARRAY = (
            env_class.STATE_IS_ARRAY if hasattr(env_class, "STATE_IS_ARRAY") else True
        )
        super(VectorizedEnvironment, self).__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_setup=delay_setup,
        )

    @property
    def n_workers(self) -> int:
        """Return the number of parallel processes that run ``step_batch`` in parallel."""
        return self._n_workers

    @property
    def plan_env(self) -> BaseEnvironment:
        """Environment that is wrapped by the current instance."""
        return self._plangym_env

    @property
    def obs_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return self.plan_env.obs_shape

    @property
    def action_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return self.plan_env.action_shape

    @property
    def gym_env(self):
        """Return the instance of the environment that is being wrapped by plangym."""
        try:
            return self.plan_env.gym_env
        except AttributeError:
            return

    def __getattr__(self, item):
        """Forward attributes to the wrapped environment."""
        return getattr(self.plan_env, item)

    @staticmethod
    def split_similar_chunks(
        vector: Union[list, numpy.ndarray],
        n_chunks: int,
    ) -> Generator[Union[list, numpy.ndarray], None, None]:
        """
        Split an indexable object into similar chunks.

        Args:
            vector: Target indexable object to be split.
            n_chunks: Number of similar chunks.

        Returns:
            Generator that returns the chunks created after splitting the target object.

        """
        chunk_size = int(numpy.ceil(len(vector) / n_chunks))
        for i in range(0, len(vector), chunk_size):
            yield vector[i : i + chunk_size]

    @classmethod
    def batch_step_data(cls, actions, states, dt, batch_size):
        """Make batches of step data to distribute across workers."""
        no_states = states is None or states[0] is None
        states = [None] * len(actions) if no_states else states
        dt = dt if isinstance(dt, numpy.ndarray) else numpy.ones(len(states), dtype=int) * dt
        states_chunks = cls.split_similar_chunks(states, n_chunks=batch_size)
        actions_chunks = cls.split_similar_chunks(actions, n_chunks=batch_size)
        dt_chunks = cls.split_similar_chunks(dt, n_chunks=batch_size)
        return states_chunks, actions_chunks, dt_chunks

    def create_env_callable(self, **kwargs) -> Callable[..., BaseEnvironment]:
        """Return a callable that initializes the environment that is being vectorized."""

        def create_env_callable(env_class, **env_kwargs):
            def _inner(**inner_kwargs):
                env_kwargs.update(inner_kwargs)
                return env_class(**env_kwargs)

            return _inner

        callable_kwargs = dict(
            env_class=self._env_class,
            name=self.name,
            frameskip=self.frameskip,
            delay_setup=self._env_class.SINGLETON,
            **self._env_kwargs,
        )
        callable_kwargs.update(kwargs)
        return create_env_callable(**callable_kwargs)

    def setup(self) -> None:
        """Initialize the target environment with the parameters provided at __init__."""
        self._plangym_env: PlanEnvironment = self.create_env_callable()()
        self._plangym_env.setup()

    def step(self, action: numpy.ndarray, state: numpy.ndarray = None, dt: int = 1):
        """
        Step the environment applying a given action from an arbitrary state.

        If is not provided the signature matches the `step` method from OpenAI gym.

        Args:
            action: Array containing the action to be applied.
            state: State to be set before stepping the environment.
            dt: Consecutive number of times to apply the given action.

        Returns:
            if states is `None` returns `(observs, rewards, ends, infos)` else
            `(new_states, observs, rewards, ends, infos)`.

        """
        return self.plan_env.step(action=action, state=state, dt=dt)

    def reset(self, return_state: bool = True):
        """
        Reset the environment and returns the first observation, or the first \
        (state, obs) tuple.

        Args:
            return_state: If true return a also the initial state of the env.

        Returns:
            Observation of the environment if `return_state` is False. Otherwise,
            return (state, obs) after reset.

        """
        state, obs = self.plan_env.reset(return_state=True)
        self.sync_states(state)
        return (state, obs) if return_state else obs

    def get_state(self):
        """
        Recover the internal state of the simulation.

        A state completely describes the Environment at a given moment.

        Returns:
            State of the simulation.

        """
        return self.plan_env.get_state()

    def set_state(self, state):
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        """
        self.plan_env.set_state(state)
        self.sync_states(state)

    def render(self, mode="human"):
        """Render the environment using OpenGL. This wraps the OpenAI render method."""
        return self.plan_env.render(mode)

    def get_image(self) -> np.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        return self.plan_env.get_image()

    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1) -> tuple:
        """
        Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        Args:
            action: Chosen action applied to the environment.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            If state is `None` returns `(observs, reward, terminal, info)`
            else returns `(new_state, observs, reward, terminal, info)`.

        """
        return self.plan_env.step_with_dt(action=action, dt=dt)

    def sample_action(self):
        """
        Return a valid action that can be used to step the Environment.

        Implementing this method is optional, and it's only intended to make the
        testing process of the Environment easier.
        """
        return self.plan_env.sample_action()

    def sync_states(self, state: None):
        """
        Synchronize the workers' states with the state of `self.gym_env`.

        Set all the states of the different workers of the internal :class:`BatchEnv`
        to the same state as the internal :class:`Environment` used to apply the
        non-vectorized steps.
        """
        raise NotImplementedError()

    def step_batch(
        self,
        actions: numpy.ndarray,
        states: numpy.ndarray = None,
        dt: [numpy.ndarray, int] = 1,
    ):
        """
        Vectorized version of the ``step`` method.

        It allows to step a vector of states and actions. The signature and
        behaviour is the same as ``step``, but taking a list of states, actions
        and dts as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            dt: int or array containing the frameskips that will be applied.

        Returns:
            if states is None returns `(observs, rewards, ends, infos)` else
            `(new_states, observs, rewards, ends, infos)`.

        """
        raise NotImplementedError()

    def clone(self, **kwargs) -> "BaseEnvironment":
        """Return a copy of the environment."""
        self_kwargs = dict(
            name=self.name,
            frameskip=self.frameskip,
            delay_setup=self.delay_setup,
            env_class=self._env_class,
            n_workers=self.n_workers,
            **self._env_kwargs,
        )
        self_kwargs.update(kwargs)
        env = self.__class__(**self_kwargs)
        return env
