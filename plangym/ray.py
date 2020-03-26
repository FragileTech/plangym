import numpy as np
import ray

from plangym.core import BaseEnvironment


def split_similar_chunks(vector: list, n_chunks: int):
    chunk_size = int(np.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i : i + chunk_size]


@ray.remote
class RemoteEnv(BaseEnvironment):
    def __init__(self, env_callable):
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
    def name(self):
        """This is the name of the environment"""
        return self.env.name

    def init_env(self):
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
        Recover the internal state of the simulation. An state must completely
        describe the Environment at a given moment.
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


class RayEnv(BaseEnvironment):
    def __init__(self, env_callable, n_workers: int, blocking: bool = False):
        self._env = env_callable()
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.blocking = blocking
        self.n_workers = n_workers
        self.workers = [RemoteEnv.remote(env_callable=env_callable) for _ in range(self.n_workers)]
        ray.get([w.init_env.remote() for w in self.workers])

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
        step = self.workers[0].step.remote(action=action, state=state, dt=dt)
        return ray.get(step) if self.blocking else step

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

    def _make_transitions(self, actions, states=None, dt: [np.ndarray, int] = None):
        states = states if states is not None else [None] * len(actions)
        if dt is None:
            dt = np.array([None] * len(states))
        dt = dt if isinstance(dt, np.ndarray) else np.ones(len(states)) * dt
        chunks = len(self.workers)
        states_chunk = split_similar_chunks(states, n_chunks=chunks)
        actions_chunk = split_similar_chunks(actions, n_chunks=chunks)
        repeat_chunk = split_similar_chunks(dt, n_chunks=chunks)
        results_ids = []
        for env, states_batch, actions_batch, dt in zip(
            self.workers, states_chunk, actions_chunk, repeat_chunk
        ):
            result = env.step_batch.remote(actions=actions_batch, states=states_batch, dt=dt)
            results_ids.append(result)
        results = ray.get(results_ids)
        _states = []
        observs = []
        rewards = []
        terminals = []
        infos = []
        for result in results:
            if states is None:
                obs, rew, ends, info = result
            else:
                _sts, obs, rew, ends, info = result
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

    def reset(self, return_state: bool = True) -> [np.ndarray, tuple]:
        """Restart the environment."""
        resets = ray.get([w.reset.remote(return_state=return_state) for w in self.workers])
        ray.get([w.set_state.remote(resets[0][0]) for w in self.workers])
        return resets[0]

    def get_state(self):
        """
        Recover the internal state of the simulation. An state must completely
        describe the Environment at a given moment.
        """
        return self._env.get_state()

    def set_state(self, state):
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None
        """
        return [w.get_state.remote(state) for w in self.workers]
