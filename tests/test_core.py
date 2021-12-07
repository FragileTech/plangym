import copy
from typing import Iterable, Tuple, Union

import gym
import numpy
import pytest

from plangym.core import BaseEnvironment
from tests.api_tests import batch_size, display, TestBaseEnvironment


class DummyEnv(BaseEnvironment):
    RETURNS_GYM_TUPLE = False
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(low=0, high=255, dtype=numpy.uint8, shape=(128,))
    dt = 1
    frameskip = 1

    @property
    def obs_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return (10,)

    @property
    def action_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return tuple()

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

    def step(
        self, action: Union[numpy.ndarray, int], state=None, dt=1
    ) -> Tuple[numpy.ndarray, ...]:
        return (1, 1, 1, False) if state is None else (self.get_state(), 1, 1, 1, False)

    def step_with_dt(self, action: Union[numpy.ndarray, int, float], dt: int = 1) -> tuple:
        return 1, 1, 1, False

    @staticmethod
    def get_lives_from_info(info):
        return info.get("lives", -1)

    def clone(self):
        return self


environments = [lambda: DummyEnv(name="dummy")]


@pytest.fixture(params=environments, scope="class")
def env(request) -> BaseEnvironment:
    return request.param()
