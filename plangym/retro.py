import sys
import traceback

from gym import spaces
import numpy as np

from plangym.core import GymEnvironment
from plangym.parallel import BatchEnv, ExternalProcess
from plangym.utils import resize_frame

try:
    import retro
except ModuleNotFoundError:
    print("Please install OpenAI retro")


class RetroEnvironment(GymEnvironment):
    """Environment for playing Atari games."""

    def __init__(
        self,
        name: str,
        dt: int = 1,
        height: float = 100,
        width: float = 100,
        wrappers=None,
        delay_init: bool = False,
        **kwargs
    ):
        self.gym_env_kwargs = kwargs
        self.height = height
        self.width = width
        super(RetroEnvironment, self).__init__(name=name, dt=dt)
        del self.gym_env
        self.gym_env = None if delay_init else self.init_env()
        if height is not None and width is not None:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
            )
        self.wrappers = wrappers

    def init_env(self):
        env = retro.make(self.name, **self.gym_env_kwargs).unwrapped
        if self.wrappers is not None:
            self.wrap_environment(self.wrappers)
        self.gym_env = env
        self.action_space = self.gym_env.action_space
        self.observation_space = (
            self.gym_env.observation_space
            if self.observation_space is None
            else self.observation_space
        )

    def __getattr__(self, item):
        return getattr(self.gym_env, item)

    def get_state(self) -> np.ndarray:
        state = self.gym_env.em.get_state()
        return np.frombuffer(state, dtype=np.int32)

    def set_state(self, state: np.ndarray):
        raw_state = state.tobytes()
        self.gym_env.em.set_state(raw_state)
        return state

    def step(self, action: np.ndarray, state: np.ndarray = None, dt: int = None) -> tuple:
        dt = dt if dt is not None else self.dt
        if state is not None:
            self.set_state(state)
        reward = 0
        for _ in range(dt):
            obs, _reward, _, info = self.gym_env.step(action)
            reward += _reward
            end_screen = info.get("screen_x", 0) >= info.get("screen_x_end", 1e6)
            terminal = info.get("x", 0) >= info.get("screen_x_end", 1e6) or end_screen
        obs = (
            resize_frame(obs, self.height, self.width)
            if self.width is not None and self.height is not None
            else obs
        )
        if state is not None:
            new_state = self.get_state()
            return new_state, obs, reward, terminal, info
        return obs, reward, terminal, info

    def step_batch(self, actions, states=None, dt: [int, np.ndarray] = None) -> tuple:
        """

        :param actions:
        :param states:
        :param dt:
        :return:
        """
        dt = dt if dt is not None else self.dt
        dt = dt if isinstance(dt, np.ndarray) else np.ones(len(states)) * dt
        data = [self.step(action, state, dt=dt) for action, state, dt in zip(actions, states, dt)]
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

    def reset(self, return_state: bool = True):
        obs = self.gym_env.reset()
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
    """Step environment in a separate process for lock free paralellism.
            The environment will be created in the external process by calling the
            specified callable. This can be an environment class, or a function
            creating the environment and potentially wrapping it. The returned
            environment should not access global variables.
       Args:
           name:
           wrappers:
           dt:
           height:
           width:
           **kwargs:

       Attributes:
          observation_space: The cached observation space of the environment.
          action_space: The cached action space of the environment.
    """

    def __init__(
        self, name, wrappers=None, dt: int = 1, height: float = 100, width: float = 100, **kwargs
    ):

        self.name = name
        super(ExternalRetro, self).__init__(
            constructor=(name, wrappers, dt, height, width, kwargs)
        )

    def _worker(self, data, conn):
        """The process waits for actions and sends back environment results.
        Args:
          data: tuple containing all the parameters for initializing a
           RetroEnvironment. This is: ( name, wrappers, dt,
           height, width, kwargs)
          conn: Connection for communication to the main process.
        Raises:
          KeyError: When receiving a message of unknown type.
        """
        try:
            name, wrappers, dt, height, width, kwargs = data
            env = RetroEnvironment(
                name, wrappers=wrappers, dt=dt, height=height, width=width, **kwargs
            )
            env.init_env()
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

            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            tf.logging.error("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class ParallelRetro(GymEnvironment):
    """Wrap any environment to be stepped in parallel.
        Args:
            name: Name of the Environment.
            dt: Frameskip that will be applied.
            height: Height of the rgb frame containing the observation.
            width: Width of the rgb frame containing the observation.
            wrappers: Wrappers to be applied to the Environment.
            n_workers: Number of workers that will be used.
            blocking: Step the environments asynchronously if False.
            **kwargs: Additional kwargs to be passed to the environment.
    """

    def __init__(
        self,
        name: str,
        dt: int = 1,
        height: float = 100,
        width: float = 100,
        wrappers=None,
        n_workers: int = 8,
        blocking: bool = False,
        **kwargs
    ):

        super(ParallelRetro, self).__init__(name=name)

        envs = [
            ExternalRetro(
                name=name, dt=dt, height=height, width=width, wrappers=wrappers, **kwargs
            )
            for _ in range(n_workers)
        ]
        self._batch_env = BatchEnv(envs, blocking)
        self.gym_env = RetroEnvironment(
            name, dt=dt, height=height, width=width, wrappers=wrappers, **kwargs
        )
        self.gym_env.init_env()
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space

    def __getattr__(self, item):
        return getattr(self.gym_env, item)

    def step_batch(
        self, actions: np.ndarray, states: np.ndarray = None, dt: [np.ndarray, int] = None,
    ):
        return self._batch_env.step_batch(actions=actions, states=states, dt=dt)

    def step(self, action: np.ndarray, state: np.ndarray = None, dt: int = None):
        return self.gym_env.step(action=action, state=state, dt=dt)

    def reset(self, return_state: bool = True, blocking: bool = True):
        state, obs = self.gym_env.reset(return_state=True)
        self.sync_states()
        return state, obs if return_state else obs

    def get_state(self):
        return self.gym_env.get_state()

    def set_state(self, state):
        self.gym_env.set_state(state)
        self.sync_states()

    def sync_states(self):
        self._batch_env.sync_states(self.get_state())
