"""Handle parallelization for ``plangym.Environment`` that allows vectorized steps."""
import atexit
import multiprocessing
import sys
import traceback
from typing import Generator, Union

import numpy

from plangym.core import BaseEnvironment, VectorizedEnvironment


def split_similar_chunks(
    vector: Union[list, numpy.ndarray],
    n_chunks: int,
) -> Generator[Union[list, numpy.ndarray], None, None]:
    """
    Split an indexable object into similar chunks.

    Args:
        vector: Target object to be split.
        n_chunks: Number of similar chunks.

    Returns:
        Generator that returns the chunks created after splitting the target object.

    """
    chunk_size = int(numpy.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i : i + chunk_size]


def batch_step_data(actions, states, dt, batch_size):
    """Make batches of step data to distribute across workers."""
    no_states = states is None or states[0] is None
    states = [None] * len(actions) if no_states else states
    dt = dt if isinstance(dt, numpy.ndarray) else numpy.ones(len(states), dtype=int) * dt
    states_chunks = split_similar_chunks(states, n_chunks=batch_size)
    actions_chunks = split_similar_chunks(actions, n_chunks=batch_size)
    dt_chunks = split_similar_chunks(dt, n_chunks=batch_size)
    return states_chunks, actions_chunks, dt_chunks


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
        if blocking:
            return promise()
        else:
            return promise

    def step_batch(self, actions, states=None, dt: [numpy.ndarray, int] = None, blocking=True):
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

        Returns:
          if states is None returns ``(observs, rewards, ends, infos)``
          else returns ``(new_states, observs, rewards, ends, infos)``

        """
        promise = self.call("step_batch", actions, states, dt)
        if blocking:
            return promise()
        else:
            return promise

    def step(self, action, state=None, dt: int = None, blocking=True):
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
        if blocking:
            return promise()
        else:
            return promise

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
        promise = self.call("reset", return_states=return_states)
        if blocking:
            return promise()
        else:
            return promise

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
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

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

            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            try:
                import tensorflow as tf

                message = f"Error in environment process: {stacktrace}"
                if hasattr(tf, "logging"):
                    tf.logging.error(message)
                else:
                    logger = tf.get_logger()
                    logger.setLevel(logging.ERROR)
                    logger.error(message)
            except ImportError:
                pass
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

    def _unpack_transitions(self, results, no_states):
        _states = []
        observs = []
        rewards = []
        terminals = []
        infos = []
        for result in results:
            if self._blocking:
                if no_states:
                    obs, rew, ends, info = result
                else:
                    _sts, obs, rew, ends, info = result
                    _states += _sts
            else:
                if no_states:
                    obs, rew, ends, info = result()
                else:
                    _sts, obs, rew, ends, info = result()
                    _states += _sts
            observs += obs
            rewards += rew
            terminals += ends
            infos += info
        if no_states:
            transitions = observs, rewards, terminals, infos
        else:
            transitions = _states, observs, rewards, terminals, infos
        return transitions

    def _make_transitions(self, actions, states=None, dt: [numpy.ndarray, int] = 1):

        results = []
        no_states = states is None or states[0] is None
        states_chunks, actions_chunks, dt_chunks = batch_step_data(
            actions=actions,
            states=states,
            dt=dt,
            batch_size=len(self._envs),
        )
        for env, states_batch, actions_batch, dt in zip(
            self._envs,
            states_chunks,
            actions_chunks,
            dt_chunks,
        ):
            result = env.step_batch(
                actions=actions_batch,
                states=states_batch,
                dt=dt,
                blocking=self._blocking,
            )
            results.append(result)
        return self._unpack_transitions(results=results, no_states=no_states)

    def step_batch(self, actions, states=None, dt: [numpy.ndarray, int] = 1):
        """
        Forward a batch of actions to the wrapped environments.

        Args:
          actions: Batched action to apply to the environment.
          states: States to be stepped. If None, act on current state.
          dt: Number of consecutive times the action will be applied.

        Raises:
          ValueError: Invalid actions.

        Returns:
          Batch of observations, rewards, and done flags.

        """
        no_states = states is None or states[0] is None
        if no_states:
            observs, rewards, dones, infos = self._make_transitions(actions, states, dt)
        else:
            states, observs, rewards, dones, infos = self._make_transitions(actions, states, dt)
        try:
            observ = numpy.stack(observs)
            reward = numpy.stack(rewards)
            done = numpy.stack(dones)
            infos = numpy.stack(infos)
        except BaseException as e:  # Lets be overconfident for once TODO: remove this.
            print(e)
            for obs in observs:
                print(obs.shape)
        if no_states:
            return observ, reward, done, infos
        else:
            return states, observs, rewards, dones, infos

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
        if self._blocking:
            observs = [self._envs[index].reset(return_states=return_states) for index in indices]
        else:
            transitions = [
                self._envs[index].reset(blocking=False, return_states=return_states)
                for index in indices
            ]
            transitions = [trans() for trans in transitions]
            states, observs = zip(*transitions)

        observ = numpy.stack(observs)
        if return_states:
            return numpy.array(states), observ
        return observ

    def close(self):
        """Send close messages to the external process and join them."""
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


class ParallelEnvironment(VectorizedEnvironment):
    """
    Allow any environment to be stepped in parallel when step_batch is called.

    It creates a local instance of the target environment to call all other methods.

    Example::

        >>> env = ParallelEnvironment(env_class=AtariEnvironment,
        >>>                           name="MsPacman-v0",
        >>>                           clone_seeds=True,
        >>>                           autoreset=True,
        >>>                           blocking=False)
        >>>
        >>> state, obs = env.reset()
        >>>
        >>> states = [state.copy() for _ in range(10)]
        >>> actions = [env.action_space.sample() for _ in range(10)]
        >>>
        >>> data =  env.step_batch(states=states,
        >>>                        actions=actions)
        >>> new_states, observs, rewards, ends, infos = data

    """

    def __init__(
        self,
        env_class,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        delay_init: bool = False,
        n_workers: int = 8,
        blocking: bool = False,
        **kwargs,
    ):
        """
        Initialize a :class:`ParallelEnvironment`.

        Args:
            env_class: Class of the environment to be wrapped.
            name: Name of the environment.
            frameskip: Number of times ``step`` will me called with the same action.
            autoreset: Ignored. Always set to True. Automatically reset the environment
                      when the OpenAI environment returns ``end = True``.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``init_env`` to be called later.
            env_callable: Callable that returns an instance of the environment \
                         that will be parallelized.
            n_workers:  Number of workers that will be used to step the env.
            blocking: Step the environments synchronously.
            *args: Additional args for the environment.
            **kwargs: Additional kwargs for the environment.

        """
        self._blocking = blocking
        self._batch_env = None
        super(ParallelEnvironment, self).__init__(
            env_class=env_class,
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_init=delay_init,
            n_workers=n_workers,
            **kwargs,
        )

    @property
    def blocking(self) -> bool:
        """If True the steps are performed sequentially."""
        return self._blocking

    def init_env(self):
        """Run environment initialization and create the subprocesses for stepping in parallel."""
        external_callable = self.create_env_callable(autoreset=True, delay_init=False)
        envs = [ExternalProcess(constructor=external_callable) for _ in range(self.n_workers)]
        self._batch_env = BatchEnv(envs, blocking=self._blocking)
        # Initialize local copy last to tolerate singletons better
        super(ParallelEnvironment, self).init_env()

    def clone(self) -> "BaseEnvironment":
        """Return a copy of the environment."""
        return super(ParallelEnvironment, self).clone(blocking=self.blocking)

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
            if states is None returns ``(observs, rewards, ends, infos)`` else \
            ``(new_states, observs, rewards, ends, infos)``

        """
        return self._batch_env.step_batch(actions=actions, states=states, dt=dt)

    def sync_states(self, state: None):
        """
        Synchronize all the copies of the wrapped environment.

        Set all the states of the different workers of the internal :class:`BatchEnv`
         to the same state as the internal :class:`Environment` used to apply the
         non-vectorized steps.
        """
        state = self.get_state() if state is None else state
        self._batch_env.sync_states(state)
