from typing import Any, Callable, Dict, Iterable, Tuple, Union

import gym
from gym.envs.registration import registry as gym_registry
import numpy


wrap_callable = Union[Callable[[], gym.Wrapper], Tuple[Callable[..., gym.Wrapper], Dict[str, Any]]]


class BaseEnvironment:
    """Inherit from this class to adapt environments to different problems."""

    action_space = None
    observation_space = None
    reward_range = None
    metadata = None

    def __init__(self, name: str):
        """
        Initialize a :class:`Environment`.

        Args:
            name: Name of the environment.

        """
        self._name = name

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

    def step(self, action: Union[numpy.ndarray, int], state=None) -> Tuple[numpy.ndarray, ...]:
        """
        Take a simulation step and make the environment evolve.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it. \
                If state is None the behaviour of this function will be the \
                same as in OpenAI gym.

        Returns:
            if states is None returns ``(observs, rewards, ends, infos)``
            else returns ``(new_states, observs, rewards, ends, infos)``

        """
        raise NotImplementedError

    def step_batch(
        self,
        actions: Union[numpy.ndarray, Iterable[Union[numpy.ndarray, int]]],
        states: Union[numpy.ndarray, Iterable] = None,
    ) -> Tuple[numpy.ndarray, ...]:
        """
        Take a step on a batch of states and actions.

        Args:
            actions: Chosen actions applied to the environment.
            states: Set the environment to the given states before stepping it.
                If state is None the behaviour of this function will be the same
                as in OpenAI gym.

        Returns:
            if states is None returns ``(observs, rewards, ends, infos)``
            else returns ``(new_states, observs, rewards, ends, infos)``

        """
        raise NotImplementedError

    def reset(
        self, return_state: bool = True
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """
        Restart the environment.

        Args:
            return_state: If ``True`` it will return the state of the environment.

        Returns:
            ``obs`` if ```return_state`` is ``True`` else return ``(state, obs)``.

        """
        raise NotImplementedError

    def get_state(self) -> numpy.ndarray:
        """
        Recover the internal state of the simulation.

        An state must completely describe the Environment at a given moment.
        """
        raise NotImplementedError

    def set_state(self, state: numpy.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        raise NotImplementedError


class GymEnvironment(BaseEnvironment):
    """Base class for implementing OpenAI ``gym`` environments in ``plangym``."""

    def __init__(
        self,
        name: str,
        dt: int = 1,
        min_dt: int = 1,
        episodic_live: bool = False,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_init: bool = False,
    ):
        """
        Initialize a :class:`GymEnvironment`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            dt: Consecutive number of times a given action will be applied.
            min_dt: Number of times an action will be applied for each ``dt``.
            episodic_live: Return ``end = True`` when losing a live.
            autoreset: Automatically reset the environment when the OpenAI environment
                      returns ``end = True``.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``init_env`` to be called later.

        """
        super(GymEnvironment, self).__init__(name=name)
        self.dt = dt
        self.min_dt = min_dt
        self._wrappers = wrappers
        self.episodic_life = episodic_live
        self.autoreset = autoreset
        self.gym_env = None
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.metadata = None
        if not delay_init:
            self.init_env()

    def init_env(self):
        """Initialize the target :class:`gym.Env` instance."""
        # Remove any undocumented wrappers
        spec = gym_registry.spec(self.name)
        if hasattr(spec, "max_episode_steps"):
            setattr(spec, "_max_episode_steps", spec.max_episode_steps)
        if hasattr(spec, "max_episode_time"):
            setattr(spec, "_max_episode_time", spec.max_episode_time)
        spec.max_episode_steps = None
        spec.max_episode_time = None
        self.gym_env = spec.make()
        if self._wrappers is not None:
            self.wrap_environment(self._wrappers)
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space
        self.reward_range = self.gym_env.reward_range
        self.metadata = self.gym_env.metadata

    def __getattr__(self, item):
        return getattr(self.gym_env, item)

    def wrap_environment(self, wrappers: Iterable[wrap_callable]):
        """Wrap the underlying OpenAI gym environment."""
        for item in wrappers:
            if isinstance(item, tuple):
                wrapper, kwargs = item
                self.gym_env = wrapper(self.gym_env, **kwargs)
            else:
                self.gym_env = item(self.gym_env)

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
        dt = dt if dt is not None else self.dt
        if state is not None:
            self.set_state(state)
        obs, reward, terminal, info = self._step_with_dt(action=action, dt=dt)
        if state is not None:
            new_state = self.get_state()
            data = new_state, obs, reward, terminal, info
        else:
            data = obs, reward, terminal, info
        if info["oob"] and self.autoreset:  # It won't reset after loosing a life
            self.gym_env.reset()
        return data

    def _step_with_dt(self, action, dt):
        reward = 0
        lost_live, terminal, oob = False, False, False
        info = {"lives": -1}
        for _ in range(int(dt)):
            for _ in range(self.min_dt):
                obs, _reward, _oob, _info = self.gym_env.step(action)
                _info["lives"] = self.get_lives_from_info(_info)
                lost_live = info["lives"] > _info["lives"] or lost_live
                oob = oob or _oob
                custom_terminal = self.custom_terminal_condition(info, _info, _oob)
                terminal = terminal or oob or custom_terminal
                terminal = terminal or lost_live if self.episodic_life else terminal
                info = _info.copy()
                reward += _reward
                if terminal:
                    break
            if terminal:
                break
        # This allows to get the original values even when using an episodic life environment
        info["terminal"] = terminal
        info["lost_live"] = lost_live
        info["oob"] = oob
        info["win"] = self.get_win_condition(info)
        return obs, reward, terminal, info

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

    def step_batch(
        self,
        actions: Union[numpy.ndarray, Iterable[Union[numpy.ndarray, int]]],
        states=None,
        dt: Union[int, numpy.ndarray] = None,
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
        dt = dt if dt is not None else self.dt
        dt = (
            dt
            if isinstance(dt, (numpy.ndarray, Iterable))
            else numpy.ones(len(actions), dtype=int) * dt
        )
        no_states = states is None or states[0] is None
        states = [None] * len(actions) if no_states else states
        data = [self.step(action, state, dt=dt) for action, state, dt in zip(actions, states, dt)]
        new_states, observs, rewards, terminals, infos = [], [], [], [], []
        for d in data:
            if no_states:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            infos.append(info)
        if no_states:
            return observs, rewards, terminals, infos
        else:
            return new_states, observs, rewards, terminals, infos

    def render(self):
        """Render the environment using OpenGL. This wraps the OpenAI render method."""
        return self.gym_env.render()
