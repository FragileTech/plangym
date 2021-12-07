import pytest

from plangym.dm_control import DMControlEnv
from plangym.parallel import ParallelEnvironment
from tests.api_tests import batch_size, display, TestBaseEnvironment, TestGymEnvironment


def walker_run():
    return DMControlEnv(name="walker-run", frameskip=3)


def parallel_dm():
    return DMControlEnv(name="cartpole-balance", frameskip=3)


environments = [walker_run, parallel_dm]


@pytest.fixture(params=environments, scope="class")
def env(request) -> DMControlEnv:
    return request.param()
