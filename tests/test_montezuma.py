import pytest

from plangym.montezuma import Montezuma, MontezumaPosLevel
from plangym.parallel import ParallelEnvironment
from tests import SKIP_ATARI_TESTS


if SKIP_ATARI_TESTS:
    pytest.skip("Atari not installed, skipping", allow_module_level=True)
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def montezuma():
    return Montezuma(clone_seeds=True, autoreset=True)


def parallel_montezuma():
    return ParallelEnvironment(env_class=Montezuma, frameskip=5, name="")


environments = [montezuma, parallel_montezuma]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Montezuma:
    return request.param()


@pytest.fixture(scope="class")
def pos_level():
    return MontezumaPosLevel(1, 100, 2, 30, 16)


class TestMontezumaPosLevel:
    def test_hash(self, pos_level):
        assert isinstance(hash(pos_level), int)

    def test_compate(self, pos_level):
        assert pos_level == MontezumaPosLevel(1, 100, 2, 30, 16)

    def test_get_state(self, pos_level):
        assert pos_level.__getstate__() == pos_level.tuple

    def test_set_state(self, pos_level):
        level, score, room, x, y = (10, 9, 8, 7, 6)
        pos_level.__setstate__((level, score, room, x, y))
        assert pos_level.tuple == (10, 9, 8, 7, 6)

    def test_repr(self, pos_level):
        assert isinstance(pos_level.__repr__(), str)
