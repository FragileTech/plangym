import pytest

from plangym.montezuma import Montezuma
from plangym.parallel import ParallelEnvironment
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def montezuma():
    return Montezuma(clone_seeds=True, autoreset=True)


def parallel_montezuma():
    return ParallelEnvironment(env_class=Montezuma, frameskip=5, name="")


environments = [montezuma, parallel_montezuma]


@pytest.fixture(params=environments, scope="class")
def env(request) -> Montezuma:
    return request.param()
