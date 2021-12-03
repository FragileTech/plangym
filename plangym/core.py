from abc import ABC
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import gym
from gym.envs.registration import registry as gym_registry
import numpy
import numpy as np


wrap_callable = Union[Callable[[], gym.Wrapper], Tuple[Callable[..., gym.Wrapper], Dict[str, Any]]]


class BaseEnvironment(ABC):
    """Inherit from this class to adapt environments to different problems."""

    STATE_IS_ARRAY = True
    RETURNS_GYM_TUPLE = True

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        delay_init: bool = False,
    ):
        """
        Initialize a :class:`Environment`.

        Args:
            name: Name of the environment.
            frameskip: Number of times ``step`` will me called with the same action.
            autoreset: Automatically reset the environment when the OpenAI environment
                      returns ``end = True``.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``init_env`` to be called later.

        """
        self._name = name
        self.frameskip = frameskip
        self.autoreset = autoreset
        self.delay_init = delay_init
        if not delay_init:
            self.init_env()

    @property
    def unwrapped(self) -> "BaseEnvironment":
        """
        Completely unwrap this Environment.

        Returns:
            plangym.Environment: The base non-wrapped plangym.Environment instance

        """
        return self

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

    def __del__(self):
        """Teardown the Environment when it is no longer needed."""
        return self.close()

    def step(
        self,
        action: Union[numpy.ndarray, int, float],
        state: numpy.ndarray = None,
        dt: int = 1,
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

        Returns:
            if state is None returns ``(observs, reward, terminal, info)``
            else returns ``(new_state, observs, reward, terminal, info)``

        """
        if state is not None:
            self.set_state(state)
        obs, reward, terminal, info = self.step_with_dt(action=action, dt=dt)
        if state is not None:
            new_state = self.get_state()
            data = new_state, obs, reward, terminal, info
        else:
            data = obs, reward, terminal, info
        if terminal and self.autoreset:
            self.reset(return_state=False)
        return data

    def step_batch(
        self,
        actions: Union[numpy.ndarray, Iterable[Union[numpy.ndarray, int]]],
        states=None,
        dt: Union[int, numpy.ndarray] = 1,
    ) -> Tuple[numpy.ndarray, ...]:
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
        dt = (
            dt
            if isinstance(dt, (numpy.ndarray, Iterable))
            else numpy.ones(len(actions), dtype=int) * dt
        )
        no_states = states is None or states[0] is None
        states = [None] * len(actions) if no_states else states
        data = [self.step(action, state, dt=dt) for action, state, dt in zip(actions, states, dt)]
        return tuple(zip(*data))

    def init_env(self) -> None:
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

    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1) -> tuple:
        """
         Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        Args:
            action: Chosen action applied to the environment.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            tuple containing ``(observs, reward, terminal, info)``.
        """
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

        An state must completely describe the Environment at a given moment.
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

    def get_image(self) -> Union[None, np.ndarray]:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
         are interpreted as RGB images with channels (Height, Width, RGB)
        """
        return None

    def clone(self) -> "BaseEnvironment":
        """Return a copy of the environment."""
        raise NotImplementedError()


class GymEnvironment(BaseEnvironment):
    """Base class for implementing OpenAI ``gym`` environments in ``plangym``."""

    action_space = None
    observation_space = None
    reward_range = None
    metadata = None

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        episodic_live: bool = False,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_init: bool = False,
    ):
        """
        Initialize a :class:`GymEnvironment`.

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

        """
        self._wrappers = wrappers
        self.episodic_life = episodic_live
        self._gym_env = None
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.metadata = None
        super(GymEnvironment, self).__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_init=delay_init,
        )

    @property
    def gym_env(self):
        """Return the instance of the environment that is being wrapped by plangym."""
        return self._gym_env

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return self.observation_space.shape

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return self.action_space.shape

    def init_env(self):
        """Initialize the target :class:`gym.Env` instance."""
        self._gym_env = self.init_gym_env()
        if self._wrappers is not None:
            self.apply_wrappers(self._wrappers)
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space
        self.reward_range = self.gym_env.reward_range
        self.metadata = self.gym_env.metadata

    def get_image(self) -> np.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
         are interpreted as RGB images with channels (Height, Width, RGB)
        """
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
        obs = self.gym_env.reset()
        return (self.get_state(), obs) if return_state else obs

    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1):
        """
         Take ``dt`` simulation steps and make the environment evolve in multiples
          of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        Args:
            action: Chosen action applied to the environment.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            if state is None returns ``(observs, reward, terminal, info)``
            else returns ``(new_state, observs, reward, terminal, info)``

        """
        reward = 0
        obs, lost_life, terminal, oob = None, False, False, False
        info = {"lives": -1}
        n_steps = 0
        for _ in range(int(dt)):
            for _ in range(self.frameskip):
                obs, _reward, _oob, _info = self.gym_env.step(action)
                _info["lives"] = self.get_lives_from_info(_info)
                lost_life = info["lives"] > _info["lives"] or lost_life
                oob = oob or _oob
                custom_terminal = self.custom_terminal_condition(info, _info, _oob)
                terminal = terminal or oob or custom_terminal
                terminal = (terminal or lost_life) if self.episodic_life else terminal
                info = _info.copy()
                reward += _reward
                n_steps += 1
                if terminal:
                    break
            if terminal:
                break
        # This allows to get the original values even when using an episodic life environment
        info["terminal"] = terminal
        info["lost_life"] = lost_life
        info["oob"] = oob
        info["win"] = self.get_win_condition(info)
        info["n_steps"] = n_steps
        return obs, reward, terminal, info

    def sample_action(self) -> Union[int, np.ndarray]:
        """Return a valid action that can be used to step the Environment chosen at random."""
        return self.action_space.sample()

    def clone(self) -> "GymEnvironment":
        """Return a copy of the environment."""
        return GymEnvironment(
            name=self.name,
            frameskip=self.frameskip,
            wrappers=self._wrappers,
            episodic_live=self.episodic_life,
            autoreset=self.autoreset,
            delay_init=self.delay_init,
        )

    def close(self):
        """Close the underlying :class:`gym.Env`."""
        return self.gym_env.close()

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`gum.Env`` instance that the current clas is wrapping."""
        # Remove any undocumented wrappers
        spec = gym_registry.spec(self.name)
        if hasattr(spec, "max_episode_steps"):
            spec._max_episode_steps = spec.max_episode_steps
        if hasattr(spec, "max_episode_time"):
            spec._max_episode_time = spec.max_episode_time
        spec.max_episode_steps = None
        spec.max_episode_time = None
        gym_env: gym.Env = spec.make()
        return gym_env

    def seed(self, seed=None):
        """Seed the underlying :class:`gym.Env`."""
        if hasattr(self.gym_env, "seed"):
            return self.gym_env.seed(seed)

    def __enter__(self):
        self.gym_env.__enter__()
        return self

    def __exit__(self):
        self.gym_env.__exit__()
        return False

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
    def get_lives_from_info(info: Dict[str, Any]) -> int:
        """Return the number of lives remaining in the current game."""
        return info.get("lives", -1)

    @staticmethod
    def get_win_condition(info: Dict[str, Any]) -> bool:
        """Return ``True`` if the current state corresponds to winning the game."""
        return False

    @staticmethod
    def custom_terminal_condition(old_info, new_info, oob) -> bool:
        """Calculate a new terminal condition using the info data."""
        return False

    def render(self, mode="human"):
        """Render the environment using OpenGL. This wraps the OpenAI render method."""
        return self.gym_env.render(mode)
