from typing import Iterable, Tuple, Union

import gym
import numpy
import pytest

from plangym.core import BaseEnvironment


class DummyEnv(BaseEnvironment):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(low=0, high=255, dtype=numpy.uint8, shape=(128,))
    dt = 1
    min_dt = 1

    def reset(self, return_state=True):
        obs = numpy.ones(10)
        return (obs, obs) if return_state else obs

    def get_state(self):
        return numpy.ones(10)

    def set_state(self, state: numpy.ndarray) -> None:
        pass

    def step_batch(
        self,
        actions: Union[numpy.ndarray, Iterable[Union[numpy.ndarray, int]]],
        states: Union[numpy.ndarray, Iterable] = None,
        dt=None,
    ) -> Tuple[numpy.ndarray, ...]:
        x = numpy.ones(len(actions))
        return (x, x, x, x) if states is None else (x, x, x, x, x)

    def step(self, action: Union[numpy.ndarray, int], state=None) -> Tuple[numpy.ndarray, ...]:
        return (1, 1, 1, False) if state is None else (1, 1, 1, 1, False)

    @staticmethod
    def get_lives_from_info(info):
        return info.get("lives", -1)


environments = [lambda: DummyEnv(name="dummy")]


@pytest.fixture(params=environments, scope="class")
def env(request) -> BaseEnvironment:
    return request.param()


class TestGymEnvironment:

    batch_size = 10

    def test_reset(self, env):
        obs = env.reset(return_state=False)
        assert isinstance(obs, (numpy.ndarray, Iterable))
        state, obs = env.reset()
        assert isinstance(state, (numpy.ndarray, Iterable))

    def test_get_state(self, env):
        env.reset()
        state = env.get_state()
        assert isinstance(state, numpy.ndarray)

    def test_set_state(self, env):
        env.reset()
        state = env.get_state()
        env.set_state(state)

    def test_action_space(self, env):
        assert hasattr(env, "action_space")
        action = env.action_space.sample()
        assert action is not None

    def test_obs_space(self, env):
        assert hasattr(env, "observation_space")

    def test_get_lives_from_info(self, env):
        info = {"lives": 3}
        lives = env.get_lives_from_info(info)
        assert lives == 3
        lives = env.get_lives_from_info({})
        assert lives == -1

    @pytest.mark.parametrize("dt", [1, 3])
    def test_step(self, env, dt):
        state, _ = env.reset()
        action = env.action_space.sample()
        data = env.step(action)
        assert isinstance(data, tuple)
        data = env.step(action, state)
        assert isinstance(data, tuple)

    @pytest.mark.parametrize("dt", [1, 3, "array"])
    def test_step_batch(self, env, dt):
        dt = dt if dt != "array" else numpy.random.randint(1, 4, self.batch_size).astype(int)
        state, _ = env.reset()
        states = [state.copy() for _ in range(self.batch_size)]
        actions = [env.action_space.sample() for _ in range(self.batch_size)]
        data = env.step_batch(actions, dt=dt)
        assert isinstance(data, tuple)
        assert len(data[0]) == self.batch_size
        data_batch = env.step_batch(actions, states)
        assert isinstance(data_batch, tuple)
        assert len(data_batch[0]) == self.batch_size
        assert len(data) == len(data_batch) - 1

    # TODO: add after finishing wrappers
    def _test_wrap_environment(self, env):
        wrappers = []
        env.wrap_environment(wrappers)
