import atexit
import multiprocessing
import sys
import traceback

from gym.envs.registration import registry as gym_registry
import numpy as np
from PIL import Image


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        height: Height of the resized image.
        width: Width of the resized image.

    Returns:
        The resized frame that matches the provided width and height.
    """
    frame = Image.fromarray(frame)
    frame = frame.convert("L").resize((height, width))
    return np.array(frame)[:, :, None]


class Environment:
    """Inherit from this class to adapt environments to different problem.
     Pretty much the same as the OpenAI gym env."""

    action_space = None
    observation_space = None
    reward_range = None
    metadata = None

    def __init__(self, name, n_repeat_action: int = 1):
        self._name = name
        self.n_repeat_action = n_repeat_action

    @property
    def unwrapped(self):
        """Completely unwrap this Environment.

        Returns:
            plangym.Environment: The base non-wrapped plangym.Environment instance
        """
        return self

    @property
    def name(self):
        """This is the name of the environment"""
        return self._name

    def step(self, action, state=None, n_repeat_action: int = 1) -> tuple:
        """
        Take a simulation step and make the environment evolve.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
                If state is None the behaviour of this function will be the
                same as in OpenAI gym.
            n_repeat_action: Consecutive number of times to apply an action.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        raise NotImplementedError

    def step_batch(
        self, actions: [np.ndarray, list], states=None, n_repeat_action: int = 1
    ) -> tuple:
        """
        Take a step on a batch of states and actions.

        Args:
            actions: Chosen actions applied to the environment.
            states: Set the environment to the given states before stepping it.
                If state is None the behaviour of this function will be the same
                as in OpenAI gym.
            n_repeat_action: Consecutive number of times that the action will be
                applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        raise NotImplementedError

    def reset(self, return_state: bool = True) -> [np.ndarray, tuple]:
        """Restart the environment."""
        raise NotImplementedError

    def get_state(self):
        """
        Recover the internal state of the simulation. An state must completely
        describe the Environment at a given moment.
        """
        raise NotImplementedError

    def set_state(self, state):
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None
        """
        raise NotImplementedError


class AtariEnvironment(Environment):
    """
    Create an environment to play OpenAI gym Atari Games.

    Args:
        name: Name of the environment. Follows standard gym syntax conventions.
        clone_seeds: Clone the random seed of the ALE emulator when
            reading/setting the state. False makes the environment stochastic.
        n_repeat_action: Consecutive number of times a given action will be applied.
        min_dt: Number of times an action will be applied for each step
            in n_repeat_action.
        obs_ram: Use ram as observations even though it is not specified in the
            `name` parameter.
        episodic_live: Return end = True when losing a live.
        autoreset: Restart environment when reaching a terminal state.

    Example::

        >>> env = AtariEnvironment(name="MsPacman-v0",
        >>>                        clone_seeds=True, autoreset=True)
        >>> state, obs = env.reset()
        >>>
        >>> states = [state.copy() for _ in range(10)]
        >>> actions = [env.action_space.sample() for _ in range(10)]
        >>>
        >>> data = env.step_batch(states=states, actions=actions)
        >>> new_states, observs, rewards, ends, infos = data

    """

    def __init__(
        self,
        name: str,
        clone_seeds: bool = True,
        n_repeat_action: int = 1,
        min_dt: int = 1,
        obs_ram: bool = False,
        episodic_live: bool = False,
        autoreset: bool = True,
    ):

        super(AtariEnvironment, self).__init__(name=name, n_repeat_action=n_repeat_action)
        self.min_dt = min_dt
        self.clone_seeds = clone_seeds
        # This is for removing undocumented wrappers.
        spec = gym_registry.spec(name)
        # not actually needed, but we feel safer
        spec.max_episode_steps = None
        spec.max_episode_time = None
        self._env = spec.make()
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata
        self.obs_ram = obs_ram
        self.episodic_life = episodic_live
        self.autoreset = autoreset

    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def n_actions(self):
        return self._env.action_space.n

    def get_state(self) -> np.ndarray:
        """
        Recover the internal state of the simulation. If clone seed is False the
        environment will be stochastic.
        Cloning the full state ensures the environment is deterministic.
        """
        if self.clone_seeds:
            return self._env.unwrapped.clone_full_state()
        else:
            return self._env.unwrapped.clone_state()

    def set_state(self, state: np.ndarray):
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None
        """
        if self.clone_seeds:
            self._env.unwrapped.restore_full_state(state)
        else:
            self._env.unwrapped.restore_state(state)
        return state

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None
    ) -> tuple:
        """

        Take n_repeat_action simulation steps and make the environment evolve
        in multiples of min_dt.
        The info dictionary will contain a boolean called 'lost_live' that will
        be true if a life was lost during the current step.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
            n_repeat_action: Consecutive number of times that the action will be applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        if state is not None:
            state = state.astype(np.uint8)
            self.set_state(state)
        reward = 0
        _end, lost_live = False, False
        info = {"lives": -1}
        terminal = False
        game_end = False
        for _ in range(int(n_repeat_action)):
            for _ in range(self.min_dt):
                obs, _reward, _end, _info = self._env.step(action)
                _info["lives"] = _info.get("ale.lives", -1)
                lost_live = info["lives"] > _info["lives"] or lost_live
                game_end = game_end or _end
                terminal = terminal or game_end
                terminal = terminal or lost_live if self.episodic_life else terminal
                info = _info.copy()
                reward += _reward
                if _end:
                    break
            if _end:
                break
        # This allows to get the original values even when using an episodic life environment
        info["terminal"] = terminal
        info["lost_live"] = lost_live
        info["game_end"] = game_end
        if self.obs_ram:
            obs = self._env.unwrapped.ale.getRAM()
        if state is not None:
            new_state = self.get_state()
            data = new_state, obs, reward, terminal, info
        else:
            data = obs, reward, terminal, info
        if _end and self.autoreset:
            self._env.reset()
        return data

    def step_batch(self, actions, states=None, n_repeat_action: [int, np.ndarray] = None) -> tuple:
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`,
        but taking a list of states, actions and n_repeat_actions as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            n_repeat_action: int or array containing the frameskips that will be applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        n_repeat_action = (
            n_repeat_action
            if isinstance(n_repeat_action, np.ndarray)
            else np.ones(len(states)) * n_repeat_action
        )
        data = [
            self.step(action, state, n_repeat_action=dt)
            for action, state, dt in zip(actions, states, n_repeat_action)
        ]
        new_states, observs, rewards, terminals, infos = [], [], [], [], []
        for d in data:
            if states is None:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            infos.append(info)
        if states is None:
            return observs, rewards, terminals, infos
        else:
            return new_states, observs, rewards, terminals, infos

    def reset(self, return_state: bool = True) -> [np.ndarray, tuple]:
        """
        Resets the environment and returns the first observation, or the first
        (state, obs) tuple.
        Args:
            return_state: If true return a also the initial state of the env.

        Returns:
            Observation of the environment if `return_state` is False. Otherwise
            return (state, obs) after reset.
        """
        if self.obs_ram:
            obs = self._env.unwrapped.ale.getRAM()
        else:
            obs = self._env.reset()
        if not return_state:
            return obs
        else:
            return self.get_state(), obs

    def render(self):
        """Render the environment using OpenGL. This wraps the OpenAI render method."""
        return self._env.render()


def split_similar_chunks(vector: list, n_chunks: int):
    chunk_size = int(np.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i : i + chunk_size]


class ExternalProcess(object):
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
        but it lets us set and read the environment state.

    """

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, constructor):

        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker, args=(constructor, conn))
        atexit.register(self.close)
        self._process.start()
        self._observ_space = None
        self._action_space = None

    @property
    def observation_space(self):
        if not self._observ_space:
            self._observ_space = self.__getattr__("observation_space")
        return self._observ_space

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = self.__getattr__("action_space")
        return self._action_space

    def __getattr__(self, name):
        """Request an attribute from the environment.
        Note that this involves communication with the external process, so it can
        be slow.

        Args:
          name: Attribute to access.

        Returns:
          Value of the attribute.
        """
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

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
        promise = self.call("set_state", state)
        if blocking:
            return promise()
        else:
            return promise

    def step_batch(
        self, actions, states=None, n_repeat_action: [np.ndarray, int] = None, blocking=True
    ):
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`, but taking
        a list of states, actions and n_repeat_actions as input.

        Args:
           actions: Iterable containing the different actions to be applied.
           states: Iterable containing the different states to be set.
           n_repeat_action: int or array containing the frameskips that will be applied.
           blocking: If True, execute sequentially.
        Returns:
          if states is None returns (observs, rewards, ends, infos)
          else returns(new_states, observs, rewards, ends, infos)
        """
        promise = self.call("step_batch", actions, states, n_repeat_action)
        if blocking:
            return promise()
        else:
            return promise

    def step(self, action, state=None, n_repeat_action: int = None, blocking=True):
        """Step the environment.

        Args:
          action: The action to apply to the environment.
          state: State to be set on the environment before stepping it.
          n_repeat_action: Number of consecutive times that action will be applied.
          blocking: Whether to wait for the result.

        Returns:
          Transition tuple when blocking, otherwise callable that returns the
          transition tuple.
        """

        promise = self.call("step", action, state, n_repeat_action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True, return_states: bool = False):
        """Reset the environment.

        Args:
          blocking: Whether to wait for the result.
          return_states: If true return also the initial state of the environment.

        Returns:
          New observation when blocking, otherwise callable that returns the new
          observation.
        """
        promise = self.call("reset", return_states=return_states)
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

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
        """The process waits for actions and sends back environment results.
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
            import tensorflow as tf
            import logging

            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            message = f"Error in environment process: {stacktrace}"
            if hasattr(tf, "logging"):
                tf.logging.error(message)
            else:
                logger = tf.get_logger()
                logger.setLevel(logging.ERROR)
                logger.error(message)
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class BatchEnv(object):
    """Combine multiple environments to step them in batch.
    It is mostly a copy paste from
    https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py
    that also allows to set and get the states.

    To step environments in parallel, environments must support a
        `blocking=False` argument to their step and reset functions that makes them
        return callables instead to receive the result at a later time.

    Args:
      envs: List of environments.
      blocking: Step environments after another rather than in parallel.

    Raises:
      ValueError: Environments have different observation or action spaces.
    """

    def __init__(self, envs, blocking):
        self._envs = envs
        self._blocking = blocking

    def __len__(self):
        """Number of combined environments."""
        return len(self._envs)

    def __getitem__(self, index):
        """Access an underlying environment by index."""
        return self._envs[index]

    def __getattr__(self, name):
        """Forward unimplemented attributes to one of the original environments.

        Args:
          name: Attribute that was accessed.

        Returns:
          Value behind the attribute name one of the wrapped environments.
        """
        return getattr(self._envs[0], name)

    def _make_transitions(self, actions, states=None, n_repeat_action: [np.ndarray, int] = None):
        states = states if states is not None else [None] * len(actions)
        if n_repeat_action is None:
            n_repeat_action = np.array([None] * len(states))
        n_repeat_action = (
            n_repeat_action
            if isinstance(n_repeat_action, np.ndarray)
            else np.ones(len(states)) * n_repeat_action
        )
        chunks = len(self._envs)
        states_chunk = split_similar_chunks(states, n_chunks=chunks)
        actions_chunk = split_similar_chunks(actions, n_chunks=chunks)
        repeat_chunk = split_similar_chunks(n_repeat_action, n_chunks=chunks)
        results = []
        for env, states_batch, actions_batch, dt in zip(
            self._envs, states_chunk, actions_chunk, repeat_chunk
        ):
            result = env.step_batch(
                actions=actions_batch,
                states=states_batch,
                n_repeat_action=dt,
                blocking=self._blocking,
            )
            results.append(result)

        _states = []
        observs = []
        rewards = []
        terminals = []
        infos = []
        for result in results:
            if self._blocking:
                if states is None:
                    obs, rew, ends, info = result
                else:
                    _sts, obs, rew, ends, info = result
                    _states += _sts
            else:
                if states is None:
                    obs, rew, ends, info = result()
                else:
                    _sts, obs, rew, ends, info = result()
                    _states += _sts
            observs += obs
            rewards += rew
            terminals += ends
            infos += info
        if states is None:
            transitions = observs, rewards, terminals, infos
        else:
            transitions = _states, observs, rewards, terminals, infos
        return transitions

    def step_batch(self, actions, states=None, n_repeat_action: [np.ndarray, int] = None):
        """Forward a batch of actions to the wrapped environments.
        Args:
          actions: Batched action to apply to the environment.
          states: States to be stepped. If None, act on current state.
          n_repeat_action: Number of consecutive times the action will be applied.

        Raises:
          ValueError: Invalid actions.

        Returns:
          Batch of observations, rewards, and done flags.
        """

        if states is None:
            observs, rewards, dones, infos = self._make_transitions(actions, None, n_repeat_action)
        else:
            states, observs, rewards, dones, infos = self._make_transitions(
                actions, states, n_repeat_action
            )
        try:
            observ = np.stack(observs)
            reward = np.stack(rewards)
            done = np.stack(dones)
            infos = np.stack(infos)
        except BaseException as e:  # Lets be overconfident for once TODO: remove this.
            print(e)
            for obs in observs:
                print(obs.shape)
        if states is None:
            return observ, reward, done, infos
        else:
            return states, observs, rewards, dones, infos

    def sync_states(self, state, blocking: bool = True):
        for env in self._envs:
            try:
                env.set_state(state, blocking=blocking)
            except EOFError:
                continue

    def reset(self, indices=None, return_states: bool = True):
        """Reset the environment and convert the resulting observation.

        Args:
          indices: The batch indices of environments to reset; defaults to all.
          return_states: return the corresponding states after reset.

        Returns:
          Batch of observations.
        """
        if indices is None:
            indices = np.arange(len(self._envs))
        if self._blocking:
            observs = [self._envs[index].reset(return_states=return_states) for index in indices]
        else:
            transitions = [
                self._envs[index].reset(blocking=False, return_states=return_states)
                for index in indices
            ]
            transitions = [trans() for trans in transitions]
            states, observs = zip(*transitions)

        observ = np.stack(observs)
        if return_states:
            return np.array(states), observ
        return observ

    def close(self):
        """Send close messages to the external process and join them."""
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


def env_callable(name, env_class, *args, **kwargs):
    def _dummy():
        return env_class(name, *args, **kwargs)

    return _dummy


class ParallelEnvironment(Environment):
    """
    Wrap any environment to be stepped in parallel when step_batch is called.

    Args:
        name:  Name of the Environment.
        env_class: Class of the environment to be wrapped.
        n_workers:  Number of workers that will be used to step the env.
        blocking: Step the environments synchronously.
        *args: Additional args for the environment.
        **kwargs: Additional kwargs for the environment.

    Example::

        >>> env = ParallelEnvironment(env_class=AtariEnvironment,
        >>>                           name="MsPacman-v0",
        >>>                           clone_seeds=True, autoreset=True,
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
        self, name, env_class, n_workers: int = 8, blocking: bool = False, *args, **kwargs
    ):
        super(ParallelEnvironment, self).__init__(name=name)
        self._env = env_callable(name, env_class, *args, **kwargs)()
        envs = [
            ExternalProcess(constructor=env_callable(name, env_class, *args, **kwargs))
            for _ in range(n_workers)
        ]
        self._batch_env = BatchEnv(envs, blocking)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step_batch(
        self,
        actions: np.ndarray,
        states: np.ndarray = None,
        n_repeat_action: [np.ndarray, int] = None,
    ):
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`,
        but taking a list of states, actions and n_repeat_actions as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            n_repeat_action: int or array containing the frameskips that will be applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos) else (new_states,
            observs, rewards, ends, infos)

        """
        return self._batch_env.step_batch(
            actions=actions, states=states, n_repeat_action=n_repeat_action
        )

    def step(self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None):
        """
        Step the environment applying a given action from an arbitrary state. If
        is not provided the signature matches the one from OpenAI gym. It allows
        to apply arbitrary boundary conditions to define custom end states in case
        the env was initialized with a "CustomDeath' object.

        Args:
            action: Array containing the action to be applied.
            state: State to be set before stepping the environment.
            n_repeat_action: Consecutive number of times to apply the given action.

        Returns:
            if states is None returns (observs, rewards, ends, infos) else (new_states,
            observs, rewards, ends, infos)
        """
        return self._env.step(action=action, state=state, n_repeat_action=n_repeat_action)

    def reset(self, return_state: bool = True, blocking: bool = True):
        """
        Resets the environment and returns the first observation, or the first
        (state, obs) tuple.

        Args:
            return_state: If true return a also the initial state of the env.
            blocking: If False, reset the environments asynchronously.

        Returns:
            Observation of the environment if `return_state` is False. Otherwise
            return (state, obs) after reset.
        """
        state, obs = self._env.reset(return_state=True)
        self.sync_states(state)
        return state, obs if return_state else obs

    def get_state(self):
        """
        Recover the internal state of the simulation. An state must completely
        describe the Environment at a given moment.

        Returns:
            State of the simulation.
        """
        return self._env.get_state()

    def set_state(self, state):
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.
        """
        self._env.set_state(state)
        self.sync_states(state)

    def sync_states(self, state: None):
        """Set all the states of the different workers of the internal :class: BatchEnv to the
        same state as the internal :class: DMControlEnv used to apply the
        non-vectorized steps.
        """
        state = self.get_state() if state is None else state
        self._batch_env.sync_states(state)
