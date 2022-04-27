import pytest


pytest.importorskip("balloon_learning_environment")
from plangym.api_tests import (  # noqa: F401
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)
from plangym.control.balloon import BalloonEnv


@pytest.fixture(
    params=generate_test_cases(["BalloonLearningEnvironment-v0"], BalloonEnv),
    scope="module",
)
def env(request) -> BalloonEnv:
    return request.param()
