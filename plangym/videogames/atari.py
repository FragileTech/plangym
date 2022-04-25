"""Implement the ``plangym`` API for Atari environments."""
from typing import Any, Dict, Iterable, Optional, Union

import gym
from gym.spaces import Space
import numpy

from plangym.core import wrap_callable
from plangym.videogames.env import VideogameEnv


def ale_to_ram(ale) -> numpy.ndarray:
    """Return the ram of the ale emulator."""
    ram_size = ale.getRAMSize()
    ram = numpy.zeros(ram_size, dtype=numpy.uint8)
    ale.getRAM(ram)
    return ram


class AtariEnv(VideogameEnv):
    """
    Create an environment to play OpenAI gym Atari Games that uses AtariALE as the emulator.

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
        full_action_space: Wheter to use the full range of possible actions
            or only those available in the game.
        render_mode: One of {None, "human", "rgb_aray"}.
        possible_to_win: It is possible to finish the Atari game without getting a
            terminal state that is not out of bounds or does not involve losing a life.
        wrappers: Wrappers that will be applied to the underlying OpenAI env.
            Every element of the iterable can be either a :class:`gym.Wrapper`
            or a tuple containing ``(gym.Wrapper, kwargs)``.
        array_state: Whether to return the state of the environment as a numpy array.
        clone_seeds: Clone the random seed of the ALE emulator when reading/setting
            the state. False makes the environment stochastic.

    Example::

        >>> env = plangym.make(name="ALE/MsPacman-v5", difficulty=2, mode=1)
        >>> state, obs = env.reset()
        >>>
        >>> states = [state.copy() for _ in range(10)]
        >>> actions = [env.action_space.sample() for _ in range(10)]
        >>>
        >>> data = env.step_batch(states=states, actions=actions)
        >>> new_states, observs, rewards, ends, infos = data

    """

    STATE_IS_ARRAY = True

    def __init__(
        self,
        name: str,
        frameskip: int = 5,
        episodic_life: bool = False,
        autoreset: bool = True,
        delay_setup: bool = False,
        remove_time_limit: bool = True,
        obs_type: str = "rgb",  # ram | rgb | grayscale
        mode: int = 0,  # game mode, see Machado et al. 2018
        difficulty: int = 0,  # game difficulty, see Machado et al. 2018
        repeat_action_probability: float = 0.0,  # Sticky action probability
        full_action_space: bool = False,  # Use all actions
        render_mode: Optional[str] = None,  # None | human | rgb_array
        possible_to_win: bool = False,
        wrappers: Iterable[wrap_callable] = None,
        array_state: bool = True,
        clone_seeds: bool = False,
        **kwargs,
    ):
        """
        Initialize a :class:`AtariEnvironment`.

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
            full_action_space: Wheter to use the full range of possible actions
                or only those available in the game.
            render_mode: One of {None, "human", "rgb_aray"}.
            possible_to_win: It is possible to finish the Atari game without getting a
                terminal state that is not out of bounds or does not involve losing a life.
            wrappers: Wrappers that will be applied to the underlying OpenAI env.
                Every element of the iterable can be either a :class:`gym.Wrapper`
                or a tuple containing ``(gym.Wrapper, kwargs)``.
            array_state: Whether to return the state of the environment as a numpy array.
            clone_seeds: Clone the random seed of the ALE emulator when reading/setting
                the state. False makes the environment stochastic.

        Example::

            >>> env = AtariEnv(name="ALE/MsPacman-v5", difficulty=2, mode=1)
            >>> type(env.gym_env)
            <class 'gym.envs.atari.environment.AtariEnv'>
            >>> state, obs = env.reset()
            >>> type(state)
            <class 'numpy.ndarray'>

        """
        self.clone_seeds = clone_seeds
        self._mode = mode
        self._difficulty = difficulty
        self._repeat_action_probability = repeat_action_probability
        self._full_action_space = full_action_space
        self.STATE_IS_ARRAY = array_state
        self.DEFAULT_OBS_TYPE = self._get_default_obs_type(name, obs_type)
        super(AtariEnv, self).__init__(
            name=name,
            frameskip=frameskip,
            episodic_life=episodic_life,
            autoreset=autoreset,
            delay_setup=delay_setup,
            remove_time_limit=remove_time_limit,
            obs_type=obs_type,  # ram | rgb | grayscale
            render_mode=render_mode,  # None | human | rgb_array
            wrappers=wrappers,
            **kwargs,
        )

    @property
    def ale(self):
        """
        Return the ``ale`` interface of the underlying :class:`gym.Env`.

        Example::

            >>> env = AtariEnv(name="ALE/MsPacman-v5", obs_type="ram")
            >>> type(env.ale)
            <class 'ale_py._ale_py.ALEInterface'>


        """
        return self.gym_env.unwrapped.ale

    @property
    def mode(self) -> int:
        """Return the selected game mode for the current environment."""
        return self._mode

    @property
    def difficulty(self) -> int:
        """Return the selected difficulty for the current environment."""
        return self._difficulty

    @property
    def repeat_action_probability(self) -> float:
        """Probability of repeating the same action after input."""
        return self._repeat_action_probability

    @property
    def full_action_space(self) -> bool:
        """If True the action space correspond to all possible actions in the Atari emulator."""
        return self._full_action_space

    @property
    def observation_space(self) -> Space:
        """Return the observation_space of the environment."""
        return self.gym_env.observation_space

    @staticmethod
    def _get_default_obs_type(name, obs_type) -> str:
        """Return the observation type of the internal Atari gym environment."""
        if "ram" in name or obs_type == "ram":
            return "ram"
        elif obs_type == "grayscale":
            return "grayscale"
        return "rgb"

    def get_lifes_from_info(self, info: Dict[str, Any]) -> int:
        """Return the number of lives remaining in the current game."""
        return info.get("ale.lives", super().get_lifes_from_info(info))

    def get_image(self) -> numpy.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Image is a three-dimensional array interpreted as an RGB image with
        channels (Height, Width, RGB). Ignores wrappers as it loads the
        screen directly from the emulator.

        Example::

            >>> env = AtariEnv(name="ALE/MsPacman-v5", obs_type="ram")
            >>> img = env.get_image()
            >>> img.shape
            (210, 160, 3)
        """
        return self.gym_env.ale.getScreenRGB()

    def get_ram(self) -> numpy.ndarray:
        """
        Return a numpy array containing the content of the emulator's RAM.

        The RAM is a vector array interpreted as the memory of the emulator.

         Example::

            >>> env = AtariEnv(name="ALE/MsPacman-v5", obs_type="grayscale")
            >>> ram = env.get_ram()
            >>> ram.shape, ram.dtype
            ((128,), dtype('uint8'))
        """
        return self.gym_env.ale.getRAM()

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`gym.Env`` instance that the Environment is wrapping."""
        # Remove any undocumented wrappers
        try:
            default_env_kwargs = dict(
                obs_type=self.obs_type,  # ram | rgb | grayscale
                frameskip=self.frameskip,  # frame skip
                mode=self._mode,  # game mode, see Machado et al. 2018
                difficulty=self.difficulty,  # game difficulty, see Machado et al. 2018
                repeat_action_probability=self.repeat_action_probability,  # Sticky action prob
                full_action_space=self.full_action_space,  # Use all actions
                render_mode=self.render_mode,  # None | human | rgb_array
            )
            default_env_kwargs.update(self._gym_env_kwargs)
            self._gym_env_kwargs = default_env_kwargs
            gym_env = super(AtariEnv, self).init_gym_env()
        except RuntimeError:
            gym_env: gym.Env = gym.make(self.name)
            gym_env.reset()
        return gym_env

    def get_state(self) -> numpy.ndarray:
        """
        Recover the internal state of the simulation.

        If clone seed is False the environment will be stochastic.
        Cloning the full state ensures the environment is deterministic.

        Example::

            >>> env = AtariEnv(name="Qbert-v0")
            >>> env.get_state() #doctest: +ELLIPSIS
            array([<ale_py._ale_py.ALEState object at 0x...>, None],
                  dtype=object)

            >>> env = AtariEnv(name="Qbert-v0", array_state=False)
            >>> env.get_state() #doctest: +ELLIPSIS
            <ale_py._ale_py.ALEState object at 0x...>

        """
        state = self.gym_env.unwrapped.clone_state(include_rng=self.clone_seeds)
        if self.STATE_IS_ARRAY:
            state = numpy.array((state, None), dtype=object)
        return state

    def set_state(self, state: numpy.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Example::

            >>> env = AtariEnv(name="Qbert-v0")
            >>> state, obs = env.reset()
            >>> new_state, obs, reward, end, info = env.step(env.sample_action(), state=state)
            >>> assert not (state == new_state).all()
            >>> env.set_state(state)
            >>> (state == env.get_state()).all()
            True
        """
        if self.STATE_IS_ARRAY:
            state = state[0]
        self.gym_env.unwrapped.restore_state(state)

    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1):
        """
        Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        Args:
            action: Chosen action applied to the environment.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            If state is `None` return ``(observs, reward, terminal, info)``
            else returns ``(new_state, observs, reward, terminal, info)``

        Example::

            >>> env = AtariEnv(name="Pong-v0")
            >>> obs = env.reset(return_state=False)
            >>> obs, reward, end, info = env.step_with_dt(env.sample_action(), dt=7)
            >>> assert not end

        """
        return super(AtariEnv, self).step_with_dt(action=action, dt=dt)

    def clone(self, **kwargs) -> "VideogameEnv":
        """Return a copy of the environment."""
        params = dict(
            mode=self.mode,
            difficulty=self.difficulty,
            repeat_action_probability=self.repeat_action_probability,
            full_action_space=self.full_action_space,
        )
        params.update(**kwargs)
        return super(VideogameEnv, self).clone(**params)


class AtariPyEnvironment(AtariEnv):
    """Create an environment to play OpenAI gym Atari Games that uses AtariPy as the emulator."""

    def get_state(self) -> numpy.ndarray:  # pragma: no cover
        """
        Recover the internal state of the simulation.

        If clone seed is False the environment will be stochastic.
        Cloning the full state ensures the environment is deterministic.
        """
        if self.clone_seeds:
            return self.gym_env.unwrapped.clone_full_state()
        else:
            return self.gym_env.unwrapped.clone_state()

    def set_state(self, state: numpy.ndarray) -> None:  # pragma: no cover
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        state = state.astype(numpy.uint8)
        if self.clone_seeds:
            self.gym_env.unwrapped.restore_full_state(state)
        else:
            self.gym_env.unwrapped.restore_state(state)

    def get_ram(self) -> numpy.ndarray:  # pragma: no cover
        """
        Return a numpy array containing the content of the emulator's RAM.

        The RAM is a vector array interpreted as the memory of the emulator.
        """
        return ale_to_ram(self.gym_env.unwrapped.ale)
