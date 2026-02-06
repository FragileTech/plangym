import operator
import sys

import numpy
import pytest

pytest.importorskip("mujoco")

from plangym.api_tests import (
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)
from plangym.control.mujoco import MujocoEnv
from plangym.environment_names import MUJOCO


# Use a subset of envs for faster testing
TEST_ENVS = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4"]


@pytest.fixture(
    params=zip(
        generate_test_cases(TEST_ENVS, MujocoEnv, n_workers_values=[None]),
        iter(TEST_ENVS * 3),
    ),
    ids=operator.itemgetter(1),
    scope="module",
)
def env(request) -> MujocoEnv:
    env = request.param[0]()
    yield env
    try:
        env.close()
    except Exception:
        pass


class TestMujoco(TestPlangymEnv):
    """Test MujocoEnv using the standard plangym test suite."""

    def test_state_recovery(self, env):
        """Test that get_state/set_state correctly preserve physics state."""
        env.reset()
        state1 = env.get_state()

        # Take some actions
        for _ in range(5):
            env.step(env.sample_action())

        # Restore state
        env.set_state(state1)
        state2 = env.get_state()

        assert numpy.allclose(state1, state2), "State not correctly restored"

    def test_mujoco_attributes(self, env):
        """Test MuJoCo-specific attributes are accessible."""
        env.reset()
        unwrapped = env.gym_env.unwrapped
        assert hasattr(unwrapped, "model")
        assert hasattr(unwrapped, "data")
        assert unwrapped.model.nq > 0  # Has position coordinates
        assert unwrapped.model.nv > 0  # Has velocity coordinates


class TestMujocoImageExtraction:
    """Test that RGB images can be extracted with non-image obs_type."""

    def test_coords_obs_with_image_extraction(self):
        """Verify get_image() returns RGB when using obs_type='coords' (default)."""
        env = MujocoEnv("Ant-v4", render_mode="rgb_array", autoreset=False)
        _state, obs, _info = env.reset()
        # Coords observation should be a 1D float array
        assert obs.ndim == 1
        assert obs.dtype in {numpy.float32, numpy.float64}
        # get_image() should return a valid RGB image
        img = env.get_image()
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == numpy.uint8
        env.close()

    def test_coords_obs_return_image(self):
        """Verify return_image=True populates info['rgb'] with obs_type='coords'."""
        env = MujocoEnv("Ant-v4", render_mode="rgb_array", return_image=True, autoreset=False)
        _state, obs, info = env.reset()
        assert "rgb" in info
        assert info["rgb"].ndim == 3
        assert info["rgb"].shape[2] == 3
        # Also check step
        obs, _reward, _term, _trunc, info = env.step(env.sample_action())
        assert obs.ndim == 1
        assert "rgb" in info
        assert info["rgb"].ndim == 3
        assert info["rgb"].shape[2] == 3
        env.close()


class TestMujocoParallel:
    """Test MujocoEnv parallel execution for planning reliability."""

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="macOS uses 'spawn' multiprocessing which can't pickle nested functions",
    )
    def test_parallel_state_restoration(self):
        """Test that parallel execution with state restoration is deterministic."""
        import plangym

        env = plangym.make("Ant-v4", n_workers=2, render_mode="rgb_array")
        state, _obs, _info = env.reset()

        # Same state + same action should produce identical results across workers
        action = env.sample_action()
        states = [state.copy() for _ in range(4)]
        actions = [action.copy() for _ in range(4)]

        new_states, _observs, rewards, _, _, _ = env.step_batch(
            actions, states=states, return_state=True
        )

        assert all(r == rewards[0] for r in rewards), "Rewards should be identical"
        assert all(numpy.allclose(s, new_states[0]) for s in new_states), (
            "States should be identical"
        )
        env.close()

    def test_state_includes_time(self):
        """Test that state captures simulation time for proper restoration."""
        from plangym.control.mujoco import MujocoEnv

        env = MujocoEnv("Ant-v4", render_mode="rgb_array", autoreset=False)
        env.reset()

        # Take some steps to advance time
        for _ in range(10):
            _, _, terminal, _, _ = env.step(env.sample_action())
            if terminal:
                env.reset()

        time_before = env.gym_env.unwrapped.data.time
        state = env.get_state()

        # Take more steps, tracking if we reset
        reset_occurred = False
        for _ in range(10):
            _, _, terminal, _, _ = env.step(env.sample_action())
            if terminal:
                reset_occurred = True
                env.reset()

        time_after_steps = env.gym_env.unwrapped.data.time
        # Only check time advance if no reset occurred
        if not reset_occurred:
            assert time_after_steps > time_before, "Time should advance"

        # Restore state
        env.set_state(state)
        time_restored = env.gym_env.unwrapped.data.time

        assert numpy.isclose(time_before, time_restored), (
            f"Time should be restored: {time_before} != {time_restored}"
        )
        env.close()
