"""Plangym API implementation."""
from abc import ABC
from typing import Callable, Generator, Tuple, Union

from gym.spaces import Space
import numpy

from plangym.core import PlanEnv, PlangymEnv


class VectorizedEnv(PlangymEnv, ABC):
    """
    Base class that defines the API for working with vectorized environments.

    A vectorized environment allows to step several copies of the environment in parallel
    when calling ``step_batch``.

    It creates a local copy of the environment that is the target of all the other
    methods of :class:`PlanEnv`. In practise, a :class:`VectorizedEnv`
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
        Initialize a :class:`VectorizedEnv`.

        Args:
            env_class: Class of the environment to be wrapped.
            name: Name of the environment.
            frameskip: Number of times ``step`` will be called with the same action.
            autoreset: Ignored. Always set to True. Automatically reset the environment
                when the OpenAI environment returns ``end = True``.
            delay_setup: If ``True`` do not initialize the :class:`gym.Environment`
                and wait for ``setup`` to be called later.
            n_workers:  Number of workers that will be used to step the env.
            **kwargs: Additional keyword arguments passed to env_class.__init__.

        """
        self._n_workers = n_workers
        self._env_class = env_class
        self._env_kwargs = kwargs
        self._plangym_env: Union[PlangymEnv, PlanEnv, None] = None
        self.SINGLETON = env_class.SINGLETON if hasattr(env_class, "SINGLETON") else False
        self.STATE_IS_ARRAY = (
            env_class.STATE_IS_ARRAY if hasattr(env_class, "STATE_IS_ARRAY") else True
        )
        super(VectorizedEnv, self).__init__(
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
    def plan_env(self) -> PlanEnv:
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
    def action_space(self) -> Space:
        """Return the action_space of the environment."""
        return self.plan_env.action_space

    @property
    def observation_space(self) -> Space:
        """Return the observation_space of the environment."""
        return self.plan_env.observation_space

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

    @staticmethod
    def unpack_transitions(results: list, return_states: bool):
        """Aggregate the results of stepping across diferent workers."""
        _states, observs, rewards, terminals, infos = [], [], [], [], []
        for result in results:
            if not return_states:
                obs, rew, ends, info = result
            else:
                _sts, obs, rew, ends, info = result
                _states += _sts

            observs += obs
            rewards += rew
            terminals += ends
            infos += info
        if not return_states:
            transitions = observs, rewards, terminals, infos
        else:
            transitions = _states, observs, rewards, terminals, infos
        return transitions

    def create_env_callable(self, **kwargs) -> Callable[..., PlanEnv]:
        """Return a callable that initializes the environment that is being vectorized."""

        def create_env_callable(env_class, **env_kwargs):
            def _inner(**inner_kwargs):
                env_kwargs.update(inner_kwargs)
                return env_class(**env_kwargs)

            return _inner

        sub_env_kwargs = dict(self._env_kwargs)
        sub_env_kwargs["render_mode"] = self.render_mode if self.render_mode != "human" else None
        callable_kwargs = dict(
            env_class=self._env_class,
            name=self.name,
            frameskip=self.frameskip,
            delay_setup=self._env_class.SINGLETON,
            **sub_env_kwargs,
        )
        callable_kwargs.update(kwargs)
        return create_env_callable(**callable_kwargs)

    def setup(self) -> None:
        """Initialize the target environment with the parameters provided at __init__."""
        self._plangym_env: PlangymEnv = self.create_env_callable()()
        self._plangym_env.setup()

    def step(
        self,
        action: numpy.ndarray,
        state: numpy.ndarray = None,
        dt: int = 1,
        return_state: bool = None,
    ):
        """
        Step the environment applying a given action from an arbitrary state.

        If is not provided the signature matches the `step` method from OpenAI gym.

        Args:
            action: Array containing the action to be applied.
            state: State to be set before stepping the environment.
            dt: Consecutive number of times to apply the given action.
            return_state: Whether to return the state in the returned tuple. \
                If None, `step` will return the state if `state` was passed as a parameter.

        Returns:
            if states is `None` returns `(observs, rewards, ends, infos)` else
            `(new_states, observs, rewards, ends, infos)`.

        """
        return self.plan_env.step(action=action, state=state, dt=dt, return_state=return_state)

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
        if self.plan_env is None and self.delay_setup:
            self.setup()
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

    def get_image(self) -> numpy.ndarray:
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

    def step_batch(
        self,
        actions: numpy.ndarray,
        states: numpy.ndarray = None,
        dt: [numpy.ndarray, int] = 1,
        return_state: bool = None,
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
            return_state: Whether to return the state in the returned tuple. \
                If None, `step` will return the state if `state` was passed as a parameter.

        Returns:
            if states is None returns `(observs, rewards, ends, infos)` else
            `(new_states, observs, rewards, ends, infos)`.

        """
        dt_is_array = (isinstance(dt, numpy.ndarray) and dt.shape) or isinstance(dt, (list, tuple))
        dt = dt if dt_is_array else numpy.ones(len(actions), dtype=int) * dt
        return self.make_transitions(actions, states, dt, return_state=return_state)

    def clone(self, **kwargs) -> "PlanEnv":
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

    def sync_states(self, state: None):
        """
        Synchronize the workers' states with the state of `self.gym_env`.

        Set all the states of the different workers of the internal :class:`BatchEnv`
        to the same state as the internal :class:`Environment` used to apply the
        non-vectorized steps.
        """
        raise NotImplementedError()

    def make_transitions(self, actions, states, dt, return_state: bool = None):
        """Implement the logic for stepping the environment in parallel."""
        raise NotImplementedError()
