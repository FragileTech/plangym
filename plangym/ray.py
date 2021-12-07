"""Implement a :class:`plangym.VectorizedEnvironment` that uses ray when calling `step_batch`."""
from typing import List

import numpy as np
import ray

from plangym.core import BaseEnvironment, VectorizedEnvironment
from plangym.parallel import batch_step_data


@ray.remote
class RemoteEnv(BaseEnvironment):
    """Remote ray Actor interface for a plangym.BaseEnvironment."""

    def __init__(self, env_callable):
        """Initialize a :class:`RemoteEnv`."""
        self._env_callable = env_callable
        self.env = None

    @property
    def unwrapped(self):
        """Completely unwrap this Environment.

        Returns:
            plangym.Environment: The base non-wrapped plangym.Environment instance
        """
        return self.env

    @property
    def name(self) -> str:
        """Return the name of the environment."""
        return self.env.name

    def init_env(self):
        """Init the wrapped environment."""
        self.env = self._env_callable()

    def step(self, action, state=None, dt: int = 1) -> tuple:
        """
        Take a simulation step and make the environment evolve.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
                If state is None the behaviour of this function will be the
                same as in OpenAI gym.
            dt: Consecutive number of times to apply an action.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        return self.env.step(action=action, state=state, dt=dt)

    def step_batch(self, actions: [np.ndarray, list], states=None, dt: int = 1) -> tuple:
        """
        Take a step on a batch of states and actions.

        Args:
            actions: Chosen actions applied to the environment.
            states: Set the environment to the given states before stepping it.
                If state is None the behaviour of this function will be the same
                as in OpenAI gym.
            dt: Consecutive number of times that the action will be
                applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        return self.env.step_batch(actions=actions, states=states, dt=dt)

    def reset(self, return_state: bool = True) -> [np.ndarray, tuple]:
        """Restart the environment."""
        return self.env.reset(return_state=return_state)

    def get_state(self):
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        return self.env.get_state()

    def set_state(self, state):
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None
        """
        return self.env.set_state(state=state)


class RayEnv(VectorizedEnvironment):
    """Use ray for taking steps in parallel when calling `step_batch`."""

    def __init__(
        self,
        env_class,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        delay_init: bool = False,
        n_workers: int = 8,
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
            *args: Additional args for the environment.
            **kwargs: Additional kwargs for the environment.

        """
        self._workers = None
        super(RayEnv, self).__init__(
            env_class=env_class,
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_init=delay_init,
            n_workers=n_workers,
            **kwargs,
        )

    @property
    def workers(self) -> List[RemoteEnv]:
        """Remote actors exposing copies of the environment."""
        return self._workers

    def init_env(self):
        """Run environment initialization and create the subprocesses for stepping in parallel."""
        env_callable = self.create_env_callable(autoreset=True, delay_init=False)
        workers = [RemoteEnv.remote(env_callable=env_callable) for _ in range(self.n_workers)]
        ray.get([w.init_env.remote() for w in workers])
        self._workers = workers
        # Initialize local copy last to tolerate singletons better
        super(RayEnv, self).init_env()

    def step_batch(self, actions: [np.ndarray, list], states=None, dt: int = 1) -> tuple:
        """
        Take a step on a batch of states and actions.

        Args:
            actions: Chosen actions applied to the environment.
            states: Set the environment to the given states before stepping it.
                If state is None the behaviour of this function will be the same
                as in OpenAI gym.
            dt: Consecutive number of times that the action will be
                applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        if states is None:
            observs, rewards, dones, infos = self._make_transitions(actions, None, dt)
        else:
            states, observs, rewards, dones, infos = self._make_transitions(actions, states, dt)
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

    @staticmethod
    def _unpack_transitions(results: list, no_states: bool):
        _states = []
        observs = []
        rewards = []
        terminals = []
        infos = []
        for result in results:
            if no_states:
                obs, rew, ends, info = result
            else:
                _sts, obs, rew, ends, info = result
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

    def _make_transitions(self, actions, states=None, dt: [np.ndarray, int] = 1):
        no_states = states is None or states[0] is None
        states_chunks, actions_chunks, dt_chunks = batch_step_data(
            actions=actions,
            states=states,
            dt=dt,
            batch_size=len(self.workers),
        )
        results_ids = []
        for env, states_batch, actions_batch, dt in zip(
            self.workers,
            states_chunks,
            actions_chunks,
            dt_chunks,
        ):
            result = env.step_batch.remote(actions=actions_batch, states=states_batch, dt=dt)
            results_ids.append(result)
        results = ray.get(results_ids)
        return self._unpack_transitions(results=results, no_states=no_states)

    def reset(self, return_state: bool = True) -> [np.ndarray, tuple]:
        """Restart the environment."""
        ray.get([w.reset.remote(return_state=return_state) for w in self.workers])
        state, obs = self.plangym_env.reset(return_state=True)
        ray.get([w.set_state.remote(state) for w in self.workers])
        return (state, obs) if return_state else obs

    def sync_states(self, state: None) -> None:
        """
        Synchronize all the copies of the wrapped environment.

        Set all the states of the different workers of the internal :class:`BatchEnv`
         to the same state as the internal :class:`Environment` used to apply the
         non-vectorized steps.
        """
        state = super().get_state() if state is None else state
        obj_ids = [w.set_state.remote(state) for w in self.workers]
        self.plangym_env.set_state(state)
        ray.get(obj_ids)
