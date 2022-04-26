import pytest

from plangym.vectorization.parallel import ParallelEnv
from plangym.videogames.montezuma import CustomMontezuma, MontezumaEnv, MontezumaPosLevel
from tests import SKIP_ATARI_TESTS


if SKIP_ATARI_TESTS:
    pytest.skip("Atari not installed, skipping", allow_module_level=True)
from plangym.api_tests import batch_size, display, TestPlanEnv, TestPlangymEnv  # noqa: F401


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
        assert pos_level.__eq__(MontezumaPosLevel(*pos_level.tuple))
        assert not pos_level == 6

    def test_get_state(self, pos_level):
        assert pos_level.__getstate__() == pos_level.tuple

    def test_set_state(self, pos_level):
        level, score, room, x, y = (10, 9, 8, 7, 6)
        pos_level.__setstate__((level, score, room, x, y))
        assert pos_level.tuple == (10, 9, 8, 7, 6)

    def test_repr(self, pos_level):
        assert isinstance(pos_level.__repr__(), str)


class TestCustomMontezuma:
    def test_make_pos(self, env):
        assert isinstance(env.gym_env.make_pos(1000, env.gym_env.pos), MontezumaPosLevel)

    def test_get_room(self):
        env = CustomMontezuma()
        env.get_room_xy(3)
        env.get_room_from_xy(0, 3)
        assert env.get_room_out_of_bounds(99, 99)
        assert not env.get_room_out_of_bounds(0, 0)

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
