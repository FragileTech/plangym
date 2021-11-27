import sys
from typing import Union


if sys.version in {(3, 6), (3, 7), (3, 8)}:
    import pytest

    from plangym.retro import ParallelRetro, RetroEnvironment
    from tests.test_core import TestGymEnvironment

    def retro_airstrike():
        return RetroEnvironment(name="Airstriker-Genesis")

    def parallel_retro():
        return ParallelRetro(name="Airstriker-Genesis", n_workers=2)

    environments = [retro_airstrike, parallel_retro]

    @pytest.fixture(params=environments, scope="class")
    def env(request) -> Union[RetroEnvironment, ParallelRetro]:
        env_ = request.param()
        yield env_
        env_.close()
