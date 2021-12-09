"""Implement the ``plangym`` API for Atari environments."""
import pickle
from typing import Any, Dict, Iterable, Optional, Union

import gym
import numpy

from plangym.core import VideogameEnvironment, wrap_callable


def ale_to_ram(ale) -> numpy.ndarray:
    """Return the ram of the ale emulator."""
    ram_size = ale.getRAMSize()
    ram = numpy.zeros(ram_size, dtype=numpy.uint8)
    ale.getRAM(ram)
    return ram


class AtariEnvironment(VideogameEnvironment):
    """
    Create an environment to play OpenAI gym Atari Games that uses AtariALE as the emulator.

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

    STATE_IS_ARRAY = True

    def __init__(
        self,
        name: str,
        frameskip: int = 5,
        episodic_live: bool = False,
        autoreset: bool = True,
        delay_init: bool = False,
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
    ):
        """
        Initialize a :class:`AtariEnvironment`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            frameskip: Number of times an action will be applied for each step \
                in dt.
            episodic_live: Return ``end = True`` when losing a life.
            autoreset: Restart environment when reaching a terminal state.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``init_env`` to be called later.
            remove_time_limit: If True, remove the time limit from the environment.
            obs_type: One of {"rgb", "ram", "gryscale"}.
            mode: Integer or string indicating the game mode, when available.
            difficulty: Difficulty level of the game, when available.
            repeat_action_probability: Repeat the last action with this probability.
            full_action_space: Wheter to use the full range of possible actions \
                              or only those available in the game.
            render_mode: One of {None, "human", "rgb_aray"}.
            possible_to_win: It is possible to finish the Atari game without \
                            getting a terminal state that is not out of bounds \
                            or doest not involve losing a life.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            array_state: Whether to return the state of the environment as a numpy array.
            clone_seeds: Clone the random seed of the ALE emulator when reading/setting \
                        the state. False makes the environment stochastic.

        """
        self._gym_env = None
        self.clone_seeds = clone_seeds
        super(AtariEnvironment, self).__init__(
            name=name,
            frameskip=frameskip,
            episodic_live=episodic_live,
            autoreset=autoreset,
            delay_init=delay_init,
            remove_time_limit=remove_time_limit,
            obs_type=obs_type,  # ram | rgb | grayscale
            mode=mode,  # game mode, see Machado et al. 2018
            difficulty=difficulty,  # game difficulty, see Machado et al. 2018
            repeat_action_probability=repeat_action_probability,  # Sticky action probability
            full_action_space=full_action_space,  # Use all actions
            render_mode=render_mode,  # None | human | rgb_array
            possible_to_win=possible_to_win,
            wrappers=wrappers,
        )
        self.STATE_IS_ARRAY = array_state

    @property
    def ale(self):
        """Return the ``ale`` interface of the underlying :class:`gym.Env`.."""
        return self.gym_env.unwrapped.ale

    def get_lives_from_info(self, info: Dict[str, Any]) -> int:
        """Return the number of lives remaining in the current game."""
        val = super().get_lives_from_info(info)
        return info.get("ale.lives", val)

    def get_win_condition(self, info: Dict[str, Any]) -> bool:
        """Return ``True`` if the current state corresponds to winning the game."""
        if not self.possible_to_win:
            return False
        return not info["lost_live"] and info["terminal"]

    def get_image(self) -> numpy.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Returns a three-dimensional array interpreted as an RGB image with
         channels (Height, Width, RGB).
        """
        return self.gym_env.ale.getScreenRGB()

    def get_ram(self) -> numpy.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Returns a three-dimensional array interpreted as an RGB image with
         channels (Height, Width, RGB).
        """
        return self.gym_env.ale.getRAM()

    def init_gym_env(self) -> gym.Env:
        """Initialize the :class:`gum.Env`` instance that the current clas is wrapping."""
        # Remove any undocumented wrappers
        gym_env = gym.make(
            self.name,
            obs_type=self.obs_type,  # ram | rgb | grayscale
            frameskip=self.frameskip,  # frame skip
            mode=self._mode,  # game mode, see Machado et al. 2018
            difficulty=self.difficulty,  # game difficulty, see Machado et al. 2018
            repeat_action_probability=self.repeat_action_probability,  # Sticky action probability
            full_action_space=self.full_action_space,  # Use all actions
            render_mode=self.render_mode,  # None | human | rgb_array
        )
        remove_time_limit = (
            self.has_time_limit
            and hasattr(gym_env, "_max_episode_steps")
            and isinstance(gym_env, gym.wrappers.time_limit.TimeLimit)
        )
        if remove_time_limit:
            max_steps = 1e100
            gym_env._max_episode_steps = max_steps
            if gym_env.spec is not None:
                gym_env.spec.max_episode_steps = None
        gym_env = gym_env.unwrapped
        gym_env.reset()
        return gym_env

    def get_state(self) -> numpy.ndarray:
        """
        Recover the internal state of the simulation.

        If clone seed is False the environment will be stochastic.
        Cloning the full state ensures the environment is deterministic.
        """
        state = self.gym_env.unwrapped.clone_state(include_rng=self.clone_seeds)
        if self.STATE_IS_ARRAY:
            state = numpy.frombuffer(pickle.dumps(state), dtype=numpy.uint8)
        return state

    def set_state(self, state: numpy.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        if self.STATE_IS_ARRAY:
            state = pickle.loads(state.tobytes())
        self.gym_env.unwrapped.restore_state(state)
        del state

    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1):
        """
         Take ``dt`` simulation steps and make the environment evolve in multiples \
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
        # This allows to get the original values even when using an episodic life environment
        info["terminal"] = terminal
        info["lost_live"] = lost_life
        info["oob"] = oob
        info["win"] = self.get_win_condition(info)
        info["n_steps"] = n_steps
        return obs, reward, terminal, info


class AtariPyEnvironment(AtariEnvironment):
    """Create an environment to play OpenAI gym Atari Games that uses AtariPy as the emulator."""

    def get_state(self) -> numpy.ndarray:
        """
        Recover the internal state of the simulation.

        If clone seed is False the environment will be stochastic.
        Cloning the full state ensures the environment is deterministic.
        """
        if self.clone_seeds:
            return self.gym_env.unwrapped.clone_full_state()
        else:
            return self.gym_env.unwrapped.clone_state()

    def set_state(self, state: numpy.ndarray) -> None:
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

    def step(
        self,
        action: Union[numpy.ndarray, int],
        state: numpy.ndarray = None,
        dt: int = 1,
    ) -> tuple:
        """
        Take ``dt`` simulation steps and make the environment evolve in multiples \
        of ``self.frameskip``.

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
        data = super(AtariPyEnvironment, self).step(action=action, state=state, dt=dt)
        if state is None:
            observ, reward, terminal, info = data
            observ = ale_to_ram(self.gym_env.unwrapped.ale) if self.obs_type == "ram" else observ
            return observ, reward, terminal, info
        else:
            state, observ, reward, terminal, info = data
            observ = ale_to_ram(self.gym_env.unwrapped.ale) if self.obs_type == "ram" else observ
            return state, observ, reward, terminal, info
