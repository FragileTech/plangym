import sys
import traceback
from typing import Any, Dict, Union

from gym import spaces
import numpy

from plangym.core import GymEnvironment
from plangym.parallel import BatchEnv, ExternalProcess
from plangym.utils import resize_frame

try:
    import retro
except ModuleNotFoundError:
    print("Please install OpenAI retro")


class RetroEnvironment(GymEnvironment):
    """Environment for playing ``gym-retro`` games."""

    def __init__(
        self,
        name: str,
        dt: int = 1,
        height: int = 100,
        width: int = 100,
        wrappers=None,
        obs_ram: bool = False,
        delay_init: bool = False,
        **kwargs
    ):
        """
        Initialize a :class:`RetroEnvironment`.

        Args:
            name: Name of the environment. Follows the retro syntax conventions.
            dt: Consecutive number of times a given action will be applied.
            height: Resize the observation to have this height.
            width: Resize the observations to have this width.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            obs_ram: Use ram as observations even though it is not specified in \
                    the ``name`` parameter.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``init_env`` to be called later.
            **kwargs: Passed to ``retro.make``.

        """
        self.gym_env_kwargs = kwargs
        self.obs_ram = obs_ram
        self.height = height
        self.width = width
        super(RetroEnvironment, self).__init__(
            name=name, dt=dt, delay_init=True, wrappers=wrappers
        )
        if not delay_init:
            self.init_env()
        if height is not None and width is not None:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=numpy.uint8
            )

    def init_env(self):
        """
        Initialize the internal retro environment and the class attributes \
        related to the environment.
        """
        env = retro.make(self.name, **self.gym_env_kwargs).unwrapped
        if self._wrappers is not None:
            self.wrap_environment(self._wrappers)
        self.gym_env = env
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
        self, action: Union[numpy.ndarray, int], state: numpy.ndarray = None, dt: int = None
    ) -> tuple:
        """
        Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.min_dt``.

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
        if state is None:
            observ, reward, terminal, info = data
            observ = self.get_state().copy() if self.obs_ram else self.process_obs(observ)
            return observ, reward, terminal, info
        else:
            state, observ, reward, terminal, info = data
            observ = state.copy() if self.obs_ram else self.process_obs(observ)
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
        obs = self.gym_env.reset()
        if self.obs_ram:
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


class ExternalRetro(ExternalProcess):
    """
    Step a :class:`RetroEnvironment` in a separate process for lock free paralellism.

    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Attributes:
      observation_space: The cached observation space of the environment.
      action_space: The cached action space of the environment.

    ..notes:
        This is mostly a copy paste from
        https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py,
        that lets us set and read the environment state.

    """

    def __init__(
        self,
        name,
        wrappers=None,
        dt: int = 1,
        height: float = 100,
        width: float = 100,
        obs_ram: bool = False,
        **kwargs,
    ):
        """
        Initialize a :class:`ExternalRetro`.

        Args:
            name: Name of the environment. Follows the retro syntax conventions.
            dt: Consecutive number of times a given action will be applied.
            height: Resize the observation to have this height.
            width: Resize the observations to have this width.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            obs_ram: Use ram as observations even though it is not specified in \
                    the ``name`` parameter.
            **kwargs: Passed to ``retro.make``.

        """
        self.name = name
        super(ExternalRetro, self).__init__(
            constructor=(name, wrappers, dt, height, width, obs_ram, kwargs)
        )

    def _worker(self, data, conn):
        """
        Wait for actions and sends back environment results.

        Args:
          data: tuple containing all the parameters for initializing a \
                RetroEnvironment. This is: ``( name, wrappers, dt, \
                height, width, obs_ram, kwargs)``
          conn: Connection for communication to the main process.

        Raises:
          KeyError: When receiving a message of unknown type.

        """
        try:
            name, wrappers, dt, height, width, obs_ram, kwargs = data
            env = RetroEnvironment(
                name,
                wrappers=wrappers,
                dt=dt,
                height=height,
                width=width,
                obs_ram=obs_ram,
                **kwargs
            )
            env.reset()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:  # pylint: disable=broad-except
            import logging

            tflogger = logging.getLogger("tensorflow").setLevel(logging.ERROR)

            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            tflogger.error("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()

    def set_state(self, state, blocking=True):
        """Set the state of the internal environment."""
        promise = self.call("set_state", state)
        if blocking:
            return promise()
        else:
            return promise


class ParallelRetro(RetroEnvironment):
    """:class:`RetroEnvironment` that performs ``step_batch`` in parallel."""

    def __init__(
        self,
        name: str,
        dt: int = 1,
        height: int = 100,
        width: int = 100,
        wrappers=None,
        obs_ram: bool = True,
        n_workers: int = 8,
        blocking: bool = False,
        delay_init: bool = False,
        **kwargs
    ):
        """
        Initialize a :class:`RetroEnvironment`.

        Args:
            name: Name of the environment. Follows the retro syntax conventions.
            dt: Consecutive number of times a given action will be applied.
            height: Resize the observation to have this height.
            width: Resize the observations to have this width.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            obs_ram: Use ram as observations even though it is not specified in \
                    the ``name`` parameter.
            n_workers:  Number of processes that will be spawned to step the environment.
            blocking: If ``True`` perform the steps sequentially. If ``False`` step \
                     the environments in parallel.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                      and the :class:`BatchEnv`, and wait for ``init_env`` to be \
                      called later.
            **kwargs: Passed to ``retro.make``.

        """
        super(ParallelRetro, self).__init__(
            name=name,
            delay_init=True,
            dt=dt,
            height=height,
            width=width,
            wrappers=wrappers,
            obs_ram=obs_ram,
            **kwargs
        )
        self._n_workers = n_workers
        self.blocking = blocking
        self._batch_env = None
        if not delay_init:
            self.init_env()

    def __getattr__(self, item):
        return getattr(self.gym_env, item)

    @property
    def n_workers(self) -> int:
        """Return the number of processes spawned for stepping the environment in parallel."""
        return self._n_workers

    def init_env(self):
        """Initialize the retro environment and the internal :class:`BatchEnv`."""
        envs = [
            ExternalRetro(
                name=self.name,
                dt=self.dt,
                height=self.height,
                width=self.width,
                wrappers=self._wrappers,
                **self.gym_env_kwargs
            )
            for _ in range(self.n_workers)
        ]
        self._batch_env = BatchEnv(envs, self.blocking)
        super(ParallelRetro, self).init_env()

    def step_batch(
        self,
        actions: numpy.ndarray,
        states: numpy.ndarray = None,
        dt: [numpy.ndarray, int] = None,
    ):
        """
        Vectorized version of the `step` method. It allows to step a vector of \
        states and actions.

        The signature and behaviour is the same as `step`, but taking a list of \
        states, actions and dts as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            dt: int or array containing the frameskips that will be applied.

        Returns:
            if states is None returns ``(observs, rewards, ends, infos)``
            else returns ``(new_states, observs, rewards, ends, infos)``

        """
        return self._batch_env.step_batch(actions=actions, states=states, dt=dt)

    def reset(self, return_state: bool = True, blocking: bool = True):
        """
        Restart the environment.

        Args:
            return_state: If ``True`` it will return the state of the environment.
            blocking: If True, execute sequentially.

        Returns:
            ``obs`` if ```return_state`` is ``True`` else return ``(state, obs)``.

        """
        state, obs = super(ParallelRetro, self).reset(return_state=True)
        self.sync_states()
        return state, obs if return_state else obs

    def set_state(self, state):
        """
        Set the state of the retro environment and synchronize the \
        :class:`BatchEnv` to the same state.
        """
        super(ParallelRetro, self).set_state(state=state)
        self.sync_states()

    def sync_states(self):
        """Set the states of the spawned processes to the same state as the retro environment."""
        self._batch_env.sync_states(self.get_state())

    def close(self):
        """Close the retro environment and the spawned processes."""
        self.gym_env.close()
        self._batch_env.close()
