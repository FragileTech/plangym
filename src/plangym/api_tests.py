import copy
from itertools import product
import os
import platform
import warnings

import gymnasium as gym
import numpy
import pytest
from pyvirtualdisplay import Display

import plangym
from plangym.core import PlanEnv, PlangymEnv
from plangym.vectorization.env import VectorizedEnv


def is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def skip_render() -> bool:
    """Return True if rendering tests should be skipped."""
    env_skip = os.getenv("SKIP_RENDER", "").lower()
    if env_skip in {"1", "true", "yes"}:
        return True
    if env_skip in {"0", "false", "no"}:
        return False
    # Auto-detect: skip rendering in WSL environments
    return is_wsl()


def generate_test_cases(
    names,
    env_class,
    n_workers_values=None,
    render_modes=None,
    obs_types=None,
    custom_tests=None,
) -> PlangymEnv:
    custom_tests = custom_tests or []
    n_workers_vals = [None] if n_workers_values is None else n_workers_values
    names = [names] if isinstance(names, str) else names
    skip_render_ = skip_render()
    available_render_modes = [None] if skip_render_ else env_class.AVAILABLE_RENDER_MODES
    available_obs_types = [None] if skip_render_ else env_class.AVAILABLE_OBS_TYPES
    render_modes = available_render_modes if render_modes is None else render_modes
    obs_types = available_obs_types if obs_types is None else obs_types
    # Check if this env class needs render for image observations
    # Environments with DEFAULT_OBS_TYPE in {"rgb", "grayscale"} can get images without render
    needs_render_for_images = env_class.DEFAULT_OBS_TYPE not in {"rgb", "grayscale"}

    i = 0
    for n_workers, obs_type, render_mode in product(
        n_workers_vals,
        obs_types,
        render_modes,
    ):
        # Skip invalid combinations: render_mode that doesn't provide RGB arrays with obs_type requiring images
        # Only for environments that need gymnasium render for images
        # In gymnasium 1.x, render_mode="human" returns None, only "rgb_array" returns images
        if (
            needs_render_for_images
            and render_mode != "rgb_array"
            and obs_type in {"rgb", "grayscale"}
        ):
            continue
        name = names[i % len(names)]
        i += 1
        if isinstance(name, tuple):
            name = "-".join(name)

        def _make_env(
            _name=name, _n_workers=n_workers, _obs_type=obs_type, _render_mode=render_mode
        ):
            return plangym.make(
                _name,
                n_workers=_n_workers,
                obs_type=_obs_type,
                render_mode=_render_mode,
            )

        yield _make_env
    yield from custom_tests


@pytest.fixture(scope="class")
def batch_size() -> int:
    return 10


@pytest.fixture(scope="module")
def display():
    os.environ["PYVIRTUALDISPLAY_DISPLAYFD"] = "0"
    display = Display(visible=False, size=(400, 400))
    display.start()
    yield display
    display.stop()


def step_tuple_test(env, obs, reward, terminal, info, dt=None):
    obs_is_array = isinstance(obs, numpy.ndarray)
    assert obs_is_array if env.OBS_IS_ARRAY else not obs_is_array
    assert obs.shape == env.obs_shape, (obs.shape, env.obs_shape)
    assert float(reward) + 1 == float(reward) + 1
    assert isinstance(terminal, bool)
    assert isinstance(info, dict)
    assert "n_step" in info
    assert info["n_step"] <= int(dt * env.frameskip), (dt, env.frameskip, info.get("n_step", 0))
    assert "dt" in info
    if dt is not None:
        assert info["dt"] == dt
    if env.return_image:
        assert "rgb" in info
        assert isinstance(info["rgb"], numpy.ndarray)


def step_batch_tuple_test(env, batch_size, observs, rewards, terminals, infos, dt):
    assert len(rewards) == batch_size
    assert len(terminals) == batch_size
    assert len(observs) == batch_size
    assert len(infos) == batch_size

    dts = dt if isinstance(dt, list | numpy.ndarray) else [dt] * batch_size
    for obs, reward, terminal, info, dt_ in zip(list(observs), rewards, terminals, infos, dts):
        step_tuple_test(env=env, obs=obs, reward=reward, terminal=terminal, info=info, dt=dt_)


class TestPlanEnv:
    CLASS_ATTRIBUTES = ("OBS_IS_ARRAY", "STATE_IS_ARRAY", "SINGLETON")
    PROPERTIES = (
        "unwrapped",
        "obs_shape",
        "action_shape",
        "name",
        "frameskip",
        "autoreset",
        "delay_setup",
        "return_image",
        "img_shape",
    )

    def test_init(self, env):
        pass

    def test_repr(self, env):
        assert str(env) == repr(env)

    # Test attributes and properties
    # ---------------------------------------------------------------------------------------------
    def test_class_attributes(self, env):
        for name in self.CLASS_ATTRIBUTES:
            assert hasattr(env.__class__, name), f"Env {env.name} does not have attribute {name}"
            isinstance(getattr(env.__class__, name), bool)

    def test_has_properties(self, env):
        for name in self.PROPERTIES:
            assert hasattr(env, name), f"Env {env.name} does not have property {name}"

    def test_name(self, env):
        assert isinstance(env.name, str)

    def test_obs_shape(self, env):
        assert hasattr(env, "obs_shape")
        assert isinstance(env.obs_shape, tuple)
        if env.obs_shape:
            for val in env.obs_shape:
                assert isinstance(val, int)
        obs, _info = env.reset(return_state=False)
        assert obs.shape == env.obs_shape, (obs.shape, env.obs_shape)
        obs, *_ = env.step(env.sample_action())
        assert obs.shape == env.obs_shape, (obs.shape, env.obs_shape)

    def test_img_shape(self, env):
        assert hasattr(env, "img_shape")
        if env.img_shape is None:
            return
        assert isinstance(env.img_shape, tuple)
        if env.img_shape:
            for val in env.img_shape:
                assert isinstance(val, int)

    def test_action_shape(self, env):
        assert hasattr(env, "action_shape")
        assert isinstance(env.action_shape, tuple)
        if env.action_shape:
            for val in env.action_shape:
                assert isinstance(val, int)

    def test_unwrapped(self, env):
        assert isinstance(env.unwrapped, PlanEnv)

    @pytest.mark.skipif(os.getenv("SKIP_RENDER", False), reason="No display in CI.")
    @pytest.mark.parametrize("return_image", [True, False])
    def test_return_image(self, env, return_image):
        assert isinstance(env.return_image, bool)
        # Skip if trying to test return_image=True with render_mode=None
        if return_image and getattr(env, "render_mode", "rgb_array") is None:
            pytest.skip("return_image=True requires render_mode='rgb_array'")
        if isinstance(env, VectorizedEnv):
            env.plan_env._return_image = return_image
        else:
            env._return_image = return_image
        _ = env.reset()
        *_, info = env.step(env.sample_action())
        if env.return_image:
            assert "rgb" in info

    # Test public API functions
    # ---------------------------------------------------------------------------------------------
    def test_sample_action(self, env):
        action = env.sample_action()
        if env.action_shape:
            assert action.shape == env.action_shape

    def test_get_state(self, env):
        state_reset, _obs, _info = env.reset()
        state = env.get_state()
        state_is_array = isinstance(state, numpy.ndarray)
        assert state_is_array if env.STATE_IS_ARRAY else not state_is_array
        if state_is_array and not env.SINGLETON:
            assert (state == state_reset).all(), f"original: {state} env: {env.get_state()}"

    def test_set_state(self, env):
        env.reset()
        state = env.get_state()
        env.step(env.sample_action())
        env.set_state(state)
        if env.STATE_IS_ARRAY:
            env_state = env.get_state()
            assert state.shape == env_state.shape
            if state.dtype is object and not env.SINGLETON:
                assert (state == env_state).all(), (state, env.get_state())

    def test_reset(self, env):
        _ = env.reset(return_state=False)
        state, obs, info = env.reset(return_state=True)
        if env.return_image:
            assert "rgb" in info
            assert isinstance(info["rgb"], numpy.ndarray)
            assert info["rgb"].shape == env.img_shape
        state_is_array = isinstance(state, numpy.ndarray)
        obs_is_array = isinstance(obs, numpy.ndarray)
        assert isinstance(info, dict), info
        assert state_is_array if env.STATE_IS_ARRAY else not state_is_array
        assert obs_is_array if env.OBS_IS_ARRAY else not obs_is_array

    @pytest.mark.parametrize("state", [None, True])
    @pytest.mark.parametrize("return_state", [None, True, False])
    def test_step(self, env, state, return_state, dt=1):
        state_, *_ = env.reset(return_state=True)
        if state is not None:
            state = state_
        action = env.sample_action()
        data = env.step(action, dt=dt, state=state, return_state=return_state)
        *new_state, obs, reward, terminal, _truncated, info = data
        assert isinstance(data, tuple)
        # Test return state works correctly
        should_return_state = state is not None if return_state is None else return_state
        if should_return_state:
            assert len(new_state) == 1
            new_state = new_state[0]
            state_is_array = isinstance(new_state, numpy.ndarray)
            assert state_is_array if env.STATE_IS_ARRAY else not state_is_array
            if state_is_array:
                assert state_.shape == new_state.shape
            if not env.SINGLETON and env.STATE_IS_ARRAY:
                curr_state = env.get_state()
                assert new_state.shape == curr_state.shape
                assert (new_state == curr_state).all(), (
                    f"original: {new_state[new_state != curr_state]} "
                    f"env: {curr_state[new_state != curr_state]}"
                )
        else:
            assert len(new_state) == 0
        step_tuple_test(env, obs, reward, terminal, info, dt=dt)

    @pytest.mark.parametrize("states", [None, True, "None_list"])
    @pytest.mark.parametrize("return_state", [None, True, False])
    def test_step_batch(self, env, states, return_state, batch_size):
        dt = 1
        state, *_ = env.reset()
        if states == "None_list":
            states = [None] * batch_size
        elif states:
            states = [copy.deepcopy(state) for _ in range(batch_size)]

        actions = [env.sample_action() for _ in range(batch_size)]

        data = env.step_batch(actions, dt=dt, states=states, return_state=return_state)
        *new_states, observs, rewards, terminals, _truncated, infos = data
        assert isinstance(data, tuple)
        # Test return state works correctly
        default_returns_state = (
            return_state is None and isinstance(states, list) and states[0] is not None
        )
        should_return_state = return_state or default_returns_state
        if should_return_state:
            assert len(new_states) == 1
            new_states = new_states[0]
            # TODO: update check when returning batch arrays is available
            assert isinstance(new_states, list)
            state_is_array = isinstance(new_states[0], numpy.ndarray)
            assert state_is_array if env.STATE_IS_ARRAY else not state_is_array
            if env.STATE_IS_ARRAY:
                assert state.shape == new_states[0].shape
        else:
            assert len(new_states) == 0, (len(new_states), should_return_state, return_state)

        step_batch_tuple_test(
            env=env,
            batch_size=batch_size,
            observs=observs,
            rewards=rewards,
            terminals=terminals,
            infos=infos,
            dt=dt,
        )

    def test_step_dt_values(self, env, dt=3, return_state=None):
        state = None
        _state, *_ = env.reset(return_state=True)
        action = env.sample_action()

        data = env.step(action, dt=dt, state=state, return_state=return_state)
        *new_state, obs, reward, terminal, _truncated, info = data
        assert isinstance(data, tuple)
        assert len(new_state) == 0
        step_tuple_test(env, obs, reward, terminal, info, dt=dt)

    @pytest.mark.parametrize("dt", [3, "array"])
    def test_step_batch_dt_values(self, env, dt, batch_size, states=None, return_state=None):
        rng = numpy.random.default_rng()
        dt = dt if dt != "array" else rng.integers(1, 4, batch_size).astype(int)
        _state, *_ = env.reset()
        actions = [env.sample_action() for _ in range(batch_size)]

        data = env.step_batch(actions, dt=dt, states=states, return_state=return_state)
        *new_states, observs, rewards, terminals, _truncated, infos = data
        assert isinstance(data, tuple)
        assert len(new_states) == 0, (len(new_states), return_state)

        step_batch_tuple_test(
            env=env,
            batch_size=batch_size,
            observs=observs,
            rewards=rewards,
            terminals=terminals,
            infos=infos,
            dt=dt,
        )

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.parametrize("delay_setup", [False, True])
    def test_clone_and_close(self, env, delay_setup):
        if not env.SINGLETON:
            env.delay_setup = delay_setup
            clone = env.clone()
            if clone.delay_setup:
                clone.reset()
            del clone

            clone = env.clone()
            if clone.delay_setup:
                clone.setup()
            clone.close()

    @pytest.mark.skipif(os.getenv("SKIP_RENDER", False), reason="No display in CI.")
    def test_get_image(self, env):
        if getattr(env, "render_mode", "rgb_array") is None:
            pytest.skip("get_image requires render_mode='rgb_array'")
        img = env.get_image()
        if img is not None:
            assert isinstance(img, numpy.ndarray)
            assert len(img.shape) == 2 or len(img.shape) == 3


class TestPlangymEnv:
    CLASS_ATTRIBUTES = ("AVAILABLE_OBS_TYPES", "DEFAULT_OBS_TYPE")
    PROPERTIES = (
        "gym_env",
        "obs_shape",
        "obs_type",
        "observation_space",
        "action_shape",
        "action_space",
        "reward_range",
        "metadata",
        "render_mode",
        "remove_time_limit",
        "name",
        "frameskip",
        "autoreset",
        "delay_setup",
        "return_image",
    )

    def test_class_attributes(self, env):
        for name in self.CLASS_ATTRIBUTES:
            assert hasattr(env.__class__, name), f"Env {env.name} does not have attribute {name}"
            isinstance(getattr(env.__class__, name), bool)

    def test_has_properties(self, env):
        for name in self.PROPERTIES:
            assert hasattr(env, name), f"Env {env.name} does not have property {name}"

    def test_obs_type(self, env):
        assert isinstance(env.obs_type, str)
        assert env.obs_type in env.AVAILABLE_OBS_TYPES
        assert env.DEFAULT_OBS_TYPE in env.AVAILABLE_OBS_TYPES, (
            str(env.DEFAULT_OBS_TYPE),
            env.AVAILABLE_OBS_TYPES,
        )

    def test_obvervation_space(self, env):
        assert hasattr(env, "observation_space")

        # assert isinstance(env.observation_space, gym.Space), (
        #    env.observation_space,
        #    env.DEFAULT_OBS_TYPE,
        # )
        assert env.observation_space.shape == env.obs_shape
        if env.observation_space.shape:
            obs, *_info = env.reset(return_state=False)
            obs_shape = env.observation_space.shape
            assert obs_shape == obs.shape, (obs_shape, obs.shape)

    def test_action_space(self, env):
        assert hasattr(env, "action_space")
        assert isinstance(env.action_space, gym.Space)
        assert env.action_space.shape == env.action_shape
        if env.action_space.shape:
            assert env.action_space.shape == env.sample_action().shape

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_gym_env(self, env):
        assert hasattr(env.gym_env, "reset")
        assert hasattr(env.gym_env, "step")
        if not isinstance(env, VectorizedEnv) and not env.SINGLETON:
            env.close()
            env.gym_env

    def test_reward_range(self, env):
        env.reward_range

    @pytest.mark.parametrize("delay_setup", [True, False])
    def test_delay_setup(self, env, delay_setup):
        if env.SINGLETON or isinstance(env, VectorizedEnv):
            return
        new_env = env.clone(delay_setup=delay_setup)
        assert new_env._gym_env is None if delay_setup else new_env._gym_env is not None
        assert env.gym_env is not None

    def test_has_metadata(self, env):
        assert hasattr(env, "metadata")

    def test_render_mode(self, env):
        assert hasattr(env, "render_mode")
        if env.render_mode is not None:
            assert isinstance(env.render_mode, str)
        # Skip assertion when skip_render() forces render_mode=None on envs that don't support it
        if skip_render() and env.render_mode is None:
            return
        assert env.render_mode in env.AVAILABLE_RENDER_MODES

    def test_remove_time_limit(self, env):
        assert isinstance(env.remove_time_limit, bool)
        if env.remove_time_limit and not env._wrappers:
            assert "TimeLimit" not in str(env.gym_env), env.gym_env

    def test_seed(self, env):
        env.seed()
        env.seed(1)

    def test_terminal(self, env):
        if env.autoreset:
            if not env.SINGLETON:
                env.setup()
            env.reset()
            if hasattr(env, "render_mode") and env.render_mode in {"human", "rgb_array"}:
                return
            env.step_with_dt(env.sample_action(), dt=1000)

    def test_render(self, env, display):
        # Use imperative skip (not decorator) so it's evaluated at runtime, not collection time
        # This is needed for pytest-xdist where collection happens in worker subprocesses
        if skip_render():
            pytest.skip("No display available (WSL or CI).")
        if getattr(env, "render_mode", "rgb_array") is None:
            pytest.skip("render requires render_mode != None")
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore")
            env.render()

    def test_wrap_environment(self, env):
        if isinstance(env, VectorizedEnv):
            return
        # Skip for envs that don't wrap gymnasium.Env (e.g., DMControlEnv wraps dm_control)
        if not isinstance(env.gym_env, gym.Env):
            return
        from gymnasium.wrappers import TransformReward  # noqa: PLC0415

        wrappers = [(TransformReward, {"func": lambda x: x})]
        env.apply_wrappers(wrappers)
        assert isinstance(env.gym_env, TransformReward)
        env._gym_env = env.gym_env.env

        wrappers = [(TransformReward, [lambda x: x])]
        env.apply_wrappers(wrappers)
        assert isinstance(env.gym_env, TransformReward)
        env._gym_env = env.gym_env.env

        wrappers = [(TransformReward, lambda x: x)]
        env.apply_wrappers(wrappers)
        assert isinstance(env.gym_env, TransformReward)
        env._gym_env = env.gym_env.env


class TestVideogameEnv:
    """Test the VideogameEnv class."""

    def test_ram(self, env):
        """Test the ram property."""
        assert hasattr(env, "get_ram")
        assert isinstance(env.get_ram(), numpy.ndarray)
