from typing import Tuple

import numpy
import pytest

from plangym.api_tests import batch_size, display, TestPlanEnv
from plangym.core import PlanEnv


class DummyPlanEnv(PlanEnv):
    _step_count = 0
    _state = None

    @property
    def obs_shape(self) -> tuple[int]:
        """Tuple containing the shape of the observations returned by the Environment."""
        return (10,)

    @property
    def action_shape(self) -> tuple[int]:
        """Tuple containing the shape of the actions applied to the Environment."""
        return ()

    def get_image(self):
        return numpy.zeros((10, 10, 3))

    def get_state(self):
        if self._state is None:
            state = numpy.ones(10)
            state[-1] = self._step_count
            self._state = state
            return state
        return self._state

    def set_state(self, state: numpy.ndarray) -> None:
        self._state = state

    def sample_action(self):
        return 0

    def apply_reset(self, **kwargs):
        self._step_count = 0
        return numpy.zeros(10), {}

    def apply_action(self, action) -> tuple:
        self._step_count += 1
        obs, reward, end, truncated, info = numpy.ones(10), 1, False, False, {}
        return obs, reward, end, truncated, info

    def clone(self):
        return self


environments = [lambda: DummyPlanEnv(name="dummy")]


@pytest.fixture(params=environments, scope="class")
def env(request) -> PlanEnv:
    return request.param()


@pytest.fixture(params=environments, scope="class")
def plangym_env(request) -> PlanEnv:
    return request.param()


class TestPrivateAPI:
    @pytest.mark.parametrize("dt", [1, 3])
    def test_step_with_dt(self, env, dt):
        _ = env.reset(return_state=False)
        action = env.sample_action()
        assert env.action_shape == numpy.array(action).shape
        data = env.step_with_dt(action, dt=dt)
        assert isinstance(data, tuple)


class TestRenderModeValidation:
    """Test that render_mode parameter is properly validated and respected."""

    def test_render_mode_none_is_respected(self):
        """Verify render_mode=None is stored correctly."""
        from plangym.control.classic_control import ClassicControl

        env = ClassicControl(name="CartPole-v1", render_mode=None)
        assert env.render_mode is None
        env.close()

    def test_render_mode_rgb_array_is_respected(self):
        """Verify render_mode='rgb_array' is stored correctly."""
        from plangym.control.classic_control import ClassicControl

        env = ClassicControl(name="CartPole-v1", render_mode="rgb_array")
        assert env.render_mode == "rgb_array"
        env.close()

    def test_invalid_render_mode_raises_error(self):
        """Verify invalid render_mode raises ValueError."""
        from plangym.control.classic_control import ClassicControl

        with pytest.raises(ValueError, match="Invalid render_mode"):
            ClassicControl(name="CartPole-v1", render_mode="invalid_mode")

    def test_return_image_with_render_mode_none_raises_error(self):
        """Verify return_image=True with render_mode=None raises ValueError."""
        from plangym.control.classic_control import ClassicControl

        with pytest.raises(ValueError, match="return_image=True requires"):
            ClassicControl(name="CartPole-v1", render_mode=None, return_image=True)

    def test_get_image_with_render_mode_none_raises_error(self):
        """Verify get_image() raises RuntimeError when render_mode=None."""
        from plangym.control.classic_control import ClassicControl

        env = ClassicControl(name="CartPole-v1", render_mode=None)
        env.reset()
        with pytest.raises(RuntimeError, match="Cannot get image when render_mode=None"):
            env.get_image()
        env.close()
