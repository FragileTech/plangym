"""Implement a :class:`plangym.VectorizedEnv` that uses ray when calling `step_batch`."""
from typing import List

import numpy


try:
    import ray
except ImportError:
    pass

from plangym.core import PlanEnv
from plangym.vectorization.env import VectorizedEnv


@ray.remote
class RemoteEnv(PlanEnv):
    """Remote ray Actor interface for a plangym.PlanEnv."""

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

    def setup(self):
        """Init the wrapped environment."""
        self.env = self._env_callable()

    def step(self, action, state=None, dt: int = 1, return_state: bool = None) -> tuple:
        """
        Take a simulation step and make the environment evolve.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
                If state is None the behaviour of this function will be the
                same as in OpenAI gym.
            dt: Consecutive number of times to apply an action.
            return_state: Whether to return the state in the returned tuple. \
                If None, `step` will return the state if `state` was passed as a parameter.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        return self.env.step(action=action, state=state, dt=dt, return_state=return_state)

    def step_batch(
        self,
        actions: [numpy.ndarray, list],
        states=None,
        dt: int = 1,
        return_state: bool = None,
    ) -> tuple:
        """
        Take a step on a batch of states and actions.

        Args:
            actions: Chosen actions applied to the environment.
            states: Set the environment to the given states before stepping it.
                If state is None the behaviour of this function will be the same
                as in OpenAI gym.
            dt: Consecutive number of times that the action will be
                applied.
            return_state: Whether to return the state in the returned tuple. \
                If None, `step` will return the state if `state` was passed as a parameter.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        return self.env.step_batch(
            actions=actions,
            states=states,
            dt=dt,
            return_state=return_state,
        )

    def reset(self, return_state: bool = True) -> [numpy.ndarray, tuple]:
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


class RayEnv(VectorizedEnv):
    """Use ray for taking steps in parallel when calling `step_batch`."""

    def __init__(
        self,
        env_class,
        name: str,
        frameskip: int = 1,
        autoreset: bool = True,
        delay_setup: bool = False,
        n_workers: int = 8,
        **kwargs,
    ):
        """
        Initialize a :class:`ParallelEnv`.

        Args:
            env_class: Class of the environment to be wrapped.
            name: Name of the environment.
            frameskip: Number of times ``step`` will me called with the same action.
            autoreset: Ignored. Always set to True. Automatically reset the environment
                      when the OpenAI environment returns ``end = True``.
            delay_setup: If ``True`` do not initialize the ``gym.Environment`` \
                     and wait for ``setup`` to be called later.
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
            delay_setup=delay_setup,
            n_workers=n_workers,
            **kwargs,
        )

    @property
    def workers(self) -> List[RemoteEnv]:
        """Remote actors exposing copies of the environment."""
        return self._workers

    def setup(self):
        """Run environment initialization and create the subprocesses for stepping in parallel."""
        env_callable = self.create_env_callable(autoreset=True, delay_setup=False)
        workers = [RemoteEnv.remote(env_callable=env_callable) for _ in range(self.n_workers)]
        ray.get([w.setup.remote() for w in workers])
        self._workers = workers
        # Initialize local copy last to tolerate singletons better
        super(RayEnv, self).setup()

    def make_transitions(
        self,
        actions,
        states=None,
        dt: [numpy.ndarray, int] = 1,
        return_state: bool = None,
    ):
        """Implement the logic for stepping the environment in parallel."""
        no_states = states is None or states[0] is None
        _return_state = ((not no_states) and return_state is None) or return_state
        chunks = self.batch_step_data(
            actions=actions,
            states=states,
            dt=dt,
            batch_size=len(self.workers),
        )
        results_ids = []
        for env, states_batch, actions_batch, dt in zip(self.workers, *chunks):
            result = env.step_batch.remote(
                actions=actions_batch,
                states=states_batch,
                dt=dt,
                return_state=return_state,
            )
            results_ids.append(result)
        results = ray.get(results_ids)
        return self.unpack_transitions(results=results, return_states=_return_state)

    def reset(self, return_state: bool = True) -> [numpy.ndarray, tuple]:
        """Restart the environment."""
        if self.plan_env is None and self.delay_setup:
            self.setup()
        ray.get([w.reset.remote(return_state=return_state) for w in self.workers])
        return super(RayEnv, self).reset(return_state=return_state)

    def sync_states(self, state: None) -> None:
        """
        Synchronize all the copies of the wrapped environment.

        Set all the states of the different workers of the internal :class:`BatchEnv`
         to the same state as the internal :class:`Environment` used to apply the
         non-vectorized steps.
        """
        state = super().get_state() if state is None else state
        obj_ids = [w.set_state.remote(state) for w in self.workers]
        self.plan_env.set_state(state)
        ray.get(obj_ids)
