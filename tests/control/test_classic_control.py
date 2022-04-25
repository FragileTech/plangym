import os

import pytest

from plangym.control.classic_control import ClassicControl
from plangym.environment_names import CLASSIC_CONTROL


if os.getenv("SKIP_RENDER", False) and str(os.getenv("SKIP_RENDER", False)).lower() != "false":
    pytest.skip("ClassicControl raises pyglet error on headless machines", allow_module_level=True)

from plangym.api_tests import (  # noqa: F401
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)


@pytest.fixture(params=generate_test_cases(CLASSIC_CONTROL, ClassicControl), scope="module")
def env(request) -> ClassicControl:
    env = request.param()
    yield env
    env.close()
