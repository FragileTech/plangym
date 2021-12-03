import copy
import os
from typing import Iterable

import numpy as np
import pytest


@pytest.fixture(scope="class")
def batch_size() -> int:
    return 10


class TestBaseEnvironment:
    def test_init(self, env):
        assert hasattr(env, "_name")
        assert hasattr(env, "min_dt")
        assert hasattr(env, "autoreset")
        assert hasattr(env, "delay_init")
        if env.delay_init:
            env.init_env()

    def test_unwrapped_does_not_crash(self, env):
        _ = env.unwrapped

    def test_reset(self, env):
        obs = env.reset(return_state=False)
        assert isinstance(obs, (np.ndarray, Iterable))
        state, obs = env.reset(return_state=True)
        assert isinstance(state, (np.ndarray, Iterable))

    def test_get_state(self, env):
        state_reset, obs = env.reset()
        state = env.get_state()
        if env.STATE_IS_ARRAY:
            assert isinstance(state, np.ndarray)
            assert (state == state_reset).all()

    def test_set_state(self, env):
        env.reset()
        state = env.get_state()
        env.set_state(state)
        env.step(env.sample_action())

    def test_action_shape(self, env):
        assert hasattr(env, "action_shape")
        assert isinstance(env.action_shape, tuple)

    def test_obs_shape(self, env):
        assert hasattr(env, "obs_shape")
        assert isinstance(env.obs_shape, tuple)

    @pytest.mark.parametrize("dt", [1, 3])
    def test_step_with_dt(self, env, dt):
        obs = env.reset(return_state=False)
        action = env.sample_action()
        assert env.action_shape == np.array(action).shape
        data = env.step_with_dt(action, dt=dt)
        assert isinstance(data, tuple)
        assert env.obs_shape == obs.shape

    @pytest.mark.parametrize("dt", [1, 3])
    def test_step(self, env, dt):
        state, _ = env.reset()
        action = env.sample_action()
        data = env.step(action, dt=dt)
        assert isinstance(data, tuple)
        state, *data = env.step(action, state)
        assert len(data) > 0
        if env.STATE_IS_ARRAY:
            assert isinstance(state, np.ndarray)
            assert (state == env.get_state()).all()

    @pytest.mark.parametrize("dt", [1, 3])
    def test_step_gym_tuple(self, env, dt):
        if env.RETURNS_GYM_TUPLE:
            state, _ = env.reset()
            action = env.sample_action()
            data = env.step(action)
            assert isinstance(data, tuple)
            observs, reward, terminal, info = data
            assert len(data) == 4
            assert isinstance(observs, np.ndarray)
            assert isinstance(float(reward), float)
            assert isinstance(bool(terminal), bool)
            assert isinstance(info, dict)
            data = env.step(action, state)
            assert isinstance(data, tuple)
            state, observs, reward, terminal, info = data
            assert len(data) == 5
            assert isinstance(observs, np.ndarray)
            assert isinstance(float(reward), float)
            assert isinstance(bool(terminal), bool)
            assert isinstance(info, dict)
            if env.STATE_IS_ARRAY:
                assert isinstance(state, np.ndarray)

    @pytest.mark.parametrize("dt", [1, 3, "array"])
    def test_step_batch(self, env, dt, batch_size):
        dt = dt if dt != "array" else np.random.randint(1, 4, batch_size).astype(int)
        state, _ = env.reset()
        states = [copy.deepcopy(state) for _ in range(batch_size)]
        actions = [env.sample_action() for _ in range(batch_size)]
        # Test dt and no states
        data = env.step_batch(actions, dt=dt)
        assert isinstance(data, tuple)
        assert len(data[0]) == batch_size
        # test states and no dt
        data_batch = env.step_batch(actions, states)
        assert isinstance(data_batch, tuple)
        assert len(data_batch[0]) == batch_size
        assert len(data) == len(data_batch) - 1
        # test states and dt
        data_batch = env.step_batch(actions, states, dt)
        assert isinstance(data_batch, tuple)
        assert len(data_batch[0]) == batch_size
        assert len(data) == len(data_batch) - 1

    def test_clone_and_close(self, env):
        clone = env.clone()
        if clone.delay_init:
            clone.init_env()
        del clone

        clone = env.clone()
        if clone.delay_init:
            clone.init_env()
        clone.close()

    @pytest.mark.skipif(bool(os.getenv("CI", False)), reason="No display in CI.")
    def test_get_image(self, env):
        img = env.get_image()
        if img is not None:
            assert isinstance(img, np.ndarray)
            assert len(img.shape) == 2 or len(img.shape) == 3


class TestGymEnvironment(TestBaseEnvironment):
    def test_action_space(self, env):
        assert hasattr(env, "action_space")
        action = env.action_space.sample()
        assert action is not None

    def test_obs_space(self, env):
        assert hasattr(env, "observation_space")

    def test_attributes(self, env):
        assert hasattr(env, "reward_range")
        assert hasattr(env, "metadata")
        assert hasattr(env, "episodic_life")
        assert hasattr(env, "gym_env")

    def test_get_lives_from_info(self, env):
        info = {"lives": 3}
        lives = env.get_lives_from_info(info)
        assert lives == 3
        lives = env.get_lives_from_info({})
        assert lives == -1

    def test_seed(self, env):
        env.seed()
        env.seed(1)

    @pytest.mark.skipif(bool(os.getenv("CI", False)), reason="No display in CI.")
    def test_render(self, env):
        if "pendulum" in env.name.lower():  # Bug in old version of gym
            return
        env.render()

    # TODO: add after finishing wrappers
    def _test_wrap_environment(self, env):
        wrappers = []
        env.apply_wrappers(wrappers)

    def _test_apply_wrappers(self, env):
        pass