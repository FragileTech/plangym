import numpy
import pytest

from plangym.vectorization.parallel import ParallelEnv
from plangym.videogames.montezuma import CustomMontezuma, MontezumaEnv, MontezumaPosLevel
from tests import SKIP_ATARI_TESTS


if SKIP_ATARI_TESTS:
    pytest.skip("Atari not installed, skipping", allow_module_level=True)
from plangym import api_tests
from plangym.api_tests import batch_size, display, TestPlangymEnv


def montezuma():
    return MontezumaEnv(obs_type="coords", autoreset=True, score_objects=True)


def montezuma_unproc():
    return MontezumaEnv(obs_type="rgb", autoreset=True, only_keys=True)


def parallel_montezuma():
    return ParallelEnv(
        env_class=MontezumaEnv,
        frameskip=5,
        name="",
        score_objects=True,
        objects_from_pixels=True,
    )


def montezuma_coords():
    return MontezumaEnv(autoreset=True, obs_type="coords")


environments = [montezuma, montezuma_unproc, parallel_montezuma, montezuma_coords]


@pytest.fixture(params=environments, scope="module")
def env(request) -> MontezumaEnv:
    env = request.param()
    yield env
    env.close()


@pytest.fixture(scope="module")
def pos_level():
    return MontezumaPosLevel(1, 100, 2, 30, 16)


class TestMontezumaPosLevel:
    def test_hash(self, pos_level):
        assert isinstance(hash(pos_level), int)

    def test_compate(self, pos_level):
        assert pos_level == MontezumaPosLevel(*pos_level.tuple)
        assert not pos_level == 6

    def test_get_state(self, pos_level):
        assert pos_level.__getstate__() == pos_level.tuple

    def test_set_state(self, pos_level):
        level, score, room, x, y = (10, 9, 8, 7, 6)
        pos_level.__setstate__((level, score, room, x, y))
        assert pos_level.tuple == (10, 9, 8, 7, 6)

    def test_repr(self, pos_level):
        assert isinstance(repr(pos_level), str)


class TestCustomMontezuma:
    def test_pos_from_unproc_state(self):
        env = CustomMontezuma(obs_type="rgb")
        obs = env.reset()
        for i in range(20):
            obs, *_ = env.step(0)
        facepix = env.get_face_pixels(obs)
        pos = env.pos_from_obs(face_pixels=facepix, obs=obs)
        assert isinstance(pos, MontezumaPosLevel)

    def test_get_objects_from_pixel(self):
        env = CustomMontezuma(obs_type="rgb")
        obs = env.reset()
        for i in range(20):
            obs, *_ = env.step(0)
        ob = env.get_objects_from_pixels(room=0, obs=obs, old_objects=[])
        assert isinstance(ob, int)

        env = CustomMontezuma(obs_type="rgb", objects_remember_rooms=True)
        obs = env.reset()
        for i in range(20):
            obs, *_ = env.step(0)
        tup = env.get_objects_from_pixels(room=0, obs=obs, old_objects=[])
        assert isinstance(tup, tuple)

    def test_get_room_xy(self):
        # Test cases for known rooms
        assert CustomMontezuma.get_room_xy(0) == (3, 0)
        assert CustomMontezuma.get_room_xy(23) == (8, 3)
        assert CustomMontezuma.get_room_xy(10) == (3, 2)

        # Test case for a room not in the pyramid
        assert CustomMontezuma.get_room_xy(24) is None
        assert CustomMontezuma.get_room_xy(-2) is None


class TestMontezuma(api_tests.TestPlanEnv):
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
                curr_state, new_state = curr_state[1:], new_state[1:]
                assert new_state.shape == curr_state.shape
                assert (new_state == curr_state).all(), (
                    f"original: {new_state[new_state != curr_state]} "
                    f"env: {curr_state[new_state != curr_state]}"
                )
        else:
            assert len(new_state) == 0
        api_tests.step_tuple_test(env, obs, reward, terminal, info, dt=dt)
