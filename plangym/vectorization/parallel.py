"""Handle parallelization for ``plangym.Environment`` that allows vectorized steps."""
import atexit
import multiprocessing
import sys
import traceback

import numpy

from plangym.core import PlanEnv
from plangym.vectorization.env import VectorizedEnv


class ExternalProcess:
    """
    Step environment in a separate process for lock free paralellism.

    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Args:
      constructor: Callable that creates and returns an OpenAI gym environment.

    Attributes:
      observation_space: The cached observation space of the environment.
      action_space: The cached action space of the environment.

    ..notes:
        This is mostly a copy paste from
        https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py,
        that lets us set and read the environment state.

    """

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, constructor):
        """
        Initialize a :class:`ExternalProcess`.

        Args:
            constructor: Callable that returns the target environment that will be parallelized.

        """
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker, args=(constructor, conn))
        atexit.register(self.close)
        self._process.start()
        self._observ_space = None
        self._action_space = None

    @property
    def observation_space(self):
        """Return the observation space of the internal environment."""
        if not self._observ_space:
            self._observ_space = self.__getattr__("observation_space")
        return self._observ_space

    @property
    def action_space(self):
        """Return the action space of the internal environment."""
        if not self._action_space:
            self._action_space = self.__getattr__("action_space")
        return self._action_space

    def __getattr__(self, name):
        """
        Request an attribute from the environment.

        Note that this involves communication with the external process, so it can \
        be slow.

        Args:
          name: Attribute to access.

        Returns:
          Value of the attribute.

        """
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """
        Asynchronously call a method of the external environment.

        Args:
          name: Name of the method to call.
          *args: Positional arguments to forward to the method.
          **kwargs: Keyword arguments to forward to the method.

        Returns:
          Promise object that blocks and provides the return value when called.

        """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def set_state(self, state, blocking=True):
        """Set the state of the internal environment."""
        promise = self.call("set_state", state)
        return promise() if blocking else promise

    def step_batch(
        self,
        actions,
        states=None,
        dt: [numpy.ndarray, int] = None,
        return_state: bool = None,
        blocking=True,
    ):
        """
        Vectorized version of the ``step`` method.

        It allows to step a vector of states and actions. The signature and \
        behaviour is the same as ``step``, but taking a list of states, actions \
        and dts as input.

        Args:
           actions: Iterable containing the different actions to be applied.
           states: Iterable containing the different states to be set.
           dt: int or array containing the frameskips that will be applied.
           blocking: If True, execute sequentially.
           return_state: Whether to return the state in the returned tuple. \
                If None, `step` will return the state if `state` was passed as a parameter.

        Returns:
          if states is None returns ``(observs, rewards, ends, infos)``
          else returns ``(new_states, observs, rewards, ends, infos)``

        """
        promise = self.call("step_batch", actions, states, dt, return_state)
        return promise() if blocking else promise

    def step(self, action, state=None, dt: int = 1, blocking=True):
        """
        Step the environment.

        Args:
          action: The action to apply to the environment.
          state: State to be set on the environment before stepping it.
          dt: Number of consecutive times that action will be applied.
          blocking: Whether to wait for the result.

        Returns:
          Transition tuple when blocking, otherwise callable that returns the \
          transition tuple.

        """
        promise = self.call("step", action, state, dt)
        return promise() if blocking else promise

    def reset(self, blocking=True, return_states: bool = False):
        """
        Reset the environment.

        Args:
          blocking: Whether to wait for the result.
          return_states: If true return also the initial state of the environment.

        Returns:
          New observation when blocking, otherwise callable that returns the new \
          observation.

        """
        promise = self.call("reset", return_state=return_states)
        return promise() if blocking else promise

    def _receive(self):
        """
        Wait for a message from the worker process and return its payload.

        Raises:
          Exception: An exception was raised inside the worker process.
          KeyError: The received message is of an unknown type.

        Returns:
          Payload object of the message.

        """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload  # pragma: no cover
            raise Exception(stacktrace)  # pragma: no cover
        if message == self._RESULT:
            return payload
        raise KeyError("Received unexpected message {}".format(message))  # pragma: no cover

    def _worker(self, constructor, conn):
        """
        Wait for actions and send back environment results.

        Args:
          constructor: Constructor for the OpenAI Gym environment.
          conn: Connection for communication to the main process.

        Raises:
          KeyError: When receiving a message of unknown type.

        """
        try:
            env = constructor()
            env.reset()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):  # pragma: no cover
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
                    break  # pragma: no cover
                raise KeyError(
                    "Received message of unknown type {}".format(message),
                )  # pragma: no cover
        except Exception:  # pragma: no cover # pylint: disable=broad-except
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class BatchEnv:
    """
    Combine multiple environments to step them in batch.

    It is mostly a copy paste from \
    https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py \
    that also allows to set and get the states.

    To step environments in parallel, environments must support a \
    ``blocking=False`` argument to their step and reset functions that \
    makes them return callables instead to receive the result at a later time.

    Args:
      envs: List of environments.
      blocking: Step environments after another rather than in parallel.

    Raises:
      ValueError: Environments have different observation or action spaces.

    """

    def __init__(self, envs, blocking):
        """
        Initialize a :class:`BatchEnv`.

        Args:
            envs: List of :class:`ExternalProcess` that contain the target environment.
            blocking: If ``True`` perform the steps sequentially. If ``False`` step \
                     the environments in parallel.

        """
        self._envs = envs
        self._blocking = blocking

    def __len__(self) -> int:
        """Return the number of combined environments."""
        return len(self._envs)

    def __getitem__(self, index):
        """Access an underlying environment by index."""
        return self._envs[index]

    def __getattr__(self, name):
        """
        Forward unimplemented attributes to one of the original environments.

        Args:
          name: Attribute that was accessed.

        Returns:
          Value behind the attribute name one of the wrapped environments.

        """
        return getattr(self._envs[0], name)

    def make_transitions(
        self,
        actions,
        states=None,
        dt: [numpy.ndarray, int] = 1,
        return_state: bool = None,
    ):
        """Implement the logic for stepping the environment in parallel."""
        results = []
        no_states = states is None or states[0] is None
        _return_state = ((not no_states) and return_state is None) or return_state
        chunks = ParallelEnv.batch_step_data(
            actions=actions,
            states=states,
            dt=dt,
            batch_size=len(self._envs),
        )
        for env, states_batch, actions_batch, dt in zip(self._envs, *chunks):
            result = env.step_batch(
                actions=actions_batch,
                states=states_batch,
                dt=dt,
                blocking=self._blocking,
                return_state=return_state,
            )
            results.append(result)
        results = [res if self._blocking else res() for res in results]
        return ParallelEnv.unpack_transitions(results=results, return_states=_return_state)

    def sync_states(self, state, blocking: bool = True) -> None:
        """
        Set the same state to all the environments that are inside an external process.

        Args:
            state: Target state to set on the environments.
            blocking: If ``True`` perform the update sequentially. If ``False`` step \
                     the environments in parallel.

        Returns:
            None.

        """
        for env in self._envs:
            try:
                env.set_state(state, blocking=blocking)
            except EOFError:
                continue

    def reset(self, indices=None, return_states: bool = True):
        """
        Reset the environment and return the resulting batch observations, \
        or batch of observations and states.

        Args:
          indices: The batch indices of environments to reset; defaults to all.
          return_states: return the corresponding states after reset.

        Returns:
          Batch of observations. If ``return_states`` is ``True`` return a tuple \
          containing ``(batch_of_observations, batch_of_states)``.

        """
        if indices is None:
            indices = numpy.arange(len(self._envs))
        trans = [
            self._envs[index].reset(return_states=return_states, blocking=self._blocking)
            for index in indices
        ]
        if not self._blocking:
            trans = [trans() for trans in trans]
        if return_states:
            states, obs = zip(*trans)
            states, obs = numpy.array(states), numpy.stack(obs)
            return states, obs
        return numpy.stack(trans)

    def close(self):
        """Send close messages to the external process and join them."""
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


class ParallelEnv(VectorizedEnv):
    """
    Allow any environment to be stepped in parallel when step_batch is called.

    It creates a local instance of the target environment to call all other methods.

    Example::

        >>> from plangym import AtariEnv
        >>> env = ParallelEnv(env_class=AtariEnv,
        ...                           name="MsPacman-v0",
        ...                           clone_seeds=True,
        ...                           autoreset=True,
        ...                           blocking=False)
        >>>
        >>> state, obs = env.reset()
        >>>
        >>> states = [state.copy() for _ in range(10)]
        >>> actions = [env.sample_action() for _ in range(10)]
        >>>
        >>> data =  env.step_batch(states=states, actions=actions)
        >>> new_states, observs, rewards, ends, infos = data

    """

    def __init__(
        self,
        env_class,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        delay_setup: bool = False,
        n_workers: int = 8,
        blocking: bool = False,
        **kwargs,
    ):
        """
        Initialize a :class:`ParallelEnv`.

        Args:
            env_class: Class of the environment to be wrapped.
            name: Name of the environment.
            frameskip: Number of times ``step`` will me called with the same action.
            autoreset: Ignored. Always set to True. Automatically reset the environment
                      when the OpenAI environment returns ``end = True``.
            delay_setup: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``setup`` to be called later.
            env_callable: Callable that returns an instance of the environment \
                         that will be parallelized.
            n_workers:  Number of workers that will be used to step the env.
            blocking: Step the environments synchronously.
            *args: Additional args for the environment.
            **kwargs: Additional kwargs for the environment.

        """
        self._blocking = blocking
        self._batch_env = None
        super(ParallelEnv, self).__init__(
            env_class=env_class,
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_setup=delay_setup,
            n_workers=n_workers,
            **kwargs,
        )

    @property
    def blocking(self) -> bool:
        """If True the steps are performed sequentially."""
        return self._blocking

    def setup(self):
        """Run environment initialization and create the subprocesses for stepping in parallel."""
        external_callable = self.create_env_callable(autoreset=True, delay_setup=False)
        envs = [ExternalProcess(constructor=external_callable) for _ in range(self.n_workers)]
        self._batch_env = BatchEnv(envs, blocking=self._blocking)
        # Initialize local copy last to tolerate singletons better
        super(ParallelEnv, self).setup()

    def clone(self, **kwargs) -> "PlanEnv":
        """Return a copy of the environment."""
        default_kwargs = dict(blocking=self.blocking)
        default_kwargs.update(kwargs)
        return super(ParallelEnv, self).clone(**default_kwargs)

    def make_transitions(
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
            if states is None returns ``(observs, rewards, ends, infos)`` else \
            ``(new_states, observs, rewards, ends, infos)``

        """
        return self._batch_env.make_transitions(
            actions=actions,
            states=states,
            dt=dt,
            return_state=return_state,
        )

    def sync_states(self, state: None):
        """
        Synchronize all the copies of the wrapped environment.

        Set all the states of the different workers of the internal :class:`BatchEnv`
         to the same state as the internal :class:`Environment` used to apply the
         non-vectorized steps.
        """
        state = self.get_state() if state is None else state
        self._batch_env.sync_states(state)

    def close(self) -> None:
        """Close the environment and the spawned processes."""
        if hasattr(self._batch_env, "close"):
            self._batch_env.close()
        if hasattr(self.gym_env, "close"):
            self.gym_env.close()
