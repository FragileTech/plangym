import os

import pytest


pytest.importorskip("balloon_learning_environment")
from src.plangym.api_tests import (
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)
from plangym.control.balloon import BalloonEnv


disable_balloon_tests = not bool(os.getenv("DISABLE_BALLOON_ENV"))
if disable_balloon_tests and str(disable_balloon_tests).lower() != "false":
    pytest.skip("balloon_learning_environment tests are disabled", allow_module_level=True)


@pytest.fixture(
    params=generate_test_cases(["BalloonLearningEnvironment-v0"], BalloonEnv),
    scope="module",
)
def env(request) -> BalloonEnv:
    return request.param()
