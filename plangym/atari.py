from typing import Any, Dict, Union

from gym import spaces
import numpy

from plangym.core import GymEnvironment
from plangym.utils import ale_to_ram


class AtariEnvironment(GymEnvironment):
    """
    Create an environment to play OpenAI gym Atari Games.

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
        dt: int = 1,
        min_dt: int = 1,
        obs_ram: bool = False,
        episodic_live: bool = False,
        autoreset: bool = True,
        possible_to_win: bool = False,
    ):
        """
        Initialize a :class:`AtariEnvironment`.

        Args:
            name: Name of the environment. Follows standard gym syntax conventions.
            clone_seeds: Clone the random seed of the ALE emulator when reading/setting \
                        the state. False makes the environment stochastic.
            dt: Consecutive number of times a given action will be applied.
            min_dt: Number of times an action will be applied for each step \
                in dt.
            obs_ram: Use ram as observations even though it is not specified in \
                    the ``name`` parameter.
            episodic_live: Return ``end = True`` when losing a live.
            autoreset: Restart environment when reaching a terminal state.
            possible_to_win: It is possible to finish the Atari game without \
                            getting a terminal state that is not out of bounds \
                            or doest not involve losing a live.

        """
        super(AtariEnvironment, self).__init__(
            name=name, dt=dt, min_dt=min_dt, episodic_live=episodic_live, autoreset=autoreset
        )
        self.clone_seeds = clone_seeds
        self.obs_ram = obs_ram
        if self.obs_ram:
            self.observation_space = spaces.Box(low=0, high=255, dtype=numpy.uint8, shape=(128,))
        self.possible_to_win = possible_to_win

    def __getattr__(self, item):
        return getattr(self.gym_env, item)

    @property
    def n_actions(self) -> int:
        """Return the number of actions available."""
        return self.gym_env.action_space.n

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

    def set_state(self, state: numpy.ndarray):
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
        data = super(AtariEnvironment, self).step(action=action, state=state, dt=dt)
        if state is None:
            observ, reward, terminal, info = data
            observ = self.gym_env.unwrapped.ale.getRAM() if self.obs_ram else observ
            return observ, reward, terminal, info
        else:
            state, observ, reward, terminal, info = data
            observ = ale_to_ram(self.gym_env.unwrapped.ale) if self.obs_ram else observ
            return state, observ, reward, terminal, info

    def get_lives_from_info(self, info: Dict[str, Any]) -> int:
        """Return the number of lives remaining in the current game."""
        val = super(AtariEnvironment, self).get_lives_from_info(info)
        return info.get("ale.lives", val)

    def get_win_condition(self, info: Dict[str, Any]) -> bool:
        """Return ``True`` if the current state corresponds to winning the game."""
        if not self.possible_to_win:
            return False
        return not info["lost_live"] and info["terminal"]

    def reset(self, return_state: bool = True) -> [numpy.ndarray, tuple]:
        """
        Reset the environment and return the first ``observation``, or the first \
        ``(state, obs)`` tuple.

        Args:
            return_state: If ``True`` return a also the initial state of the env.

        Returns:
            ``Observation`` of the environment if `return_state` is ``False``. \
            Otherwise return ``(state, obs)`` after reset.

        """
        obs = ale_to_ram(self.gym_env.unwrapped.ale) if self.obs_ram else self.gym_env.reset()
        if not return_state:
            return obs
        else:
            return self.get_state(), obs
