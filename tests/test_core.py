from typing import Tuple

import numpy
import pytest

from plangym.api_tests import batch_size, display, TestPlanEnvironment  # noqa: F401
from plangym.core import PlanEnvironment


class DummyPlanEnvironment(PlanEnvironment):
    _step_count = 0

    @property
    def obs_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return (10,)

    @property
    def action_shape(self) -> Tuple[int]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return tuple()

    def get_image(self):
        return numpy.zeros((10, 10, 3))

    def get_state(self):
        state = numpy.ones(10)
        state[-1] = self._step_count
        return state

    def set_state(self, state: numpy.ndarray) -> None:
        pass

    def sample_action(self):
        return 0

    def apply_reset(self, **kwargs):
        self._step_count = 0
        return numpy.zeros(10)

    def apply_action(self, action) -> tuple:
        self._step_count += 1
        obs, reward, end, info = numpy.ones(10), 1, False, {}
        return obs, reward, end, info

    def clone(self):
        return self


environments = [lambda: DummyPlanEnvironment(name="dummy")]


@pytest.fixture(params=environments, scope="class")
def env(request) -> PlanEnvironment:
    return request.param()


class TestPrivateAPI:
    @pytest.mark.parametrize("dt", [1, 3])
    def test_step_with_dt(self, env, dt):
        _ = env.reset(return_state=False)
        action = env.sample_action()
        assert env.action_shape == numpy.array(action).shape
        data = env.step_with_dt(action, dt=dt)
        assert isinstance(data, tuple)
