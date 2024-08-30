import os

import pytest

from plangym.control.classic_control import ClassicControl
from plangym.environment_names import CLASSIC_CONTROL


if (
    os.getenv("SKIP_CLASSIC_CONTROL", None)
    and str(os.getenv("SKIP_CLASSIC_CONTROL", "false")).lower() != "false"
):
    pytest.skip("Skipping classic control", allow_module_level=True)

from plangym.api_tests import (
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)
import operator


@pytest.fixture(
    params=zip(generate_test_cases(CLASSIC_CONTROL, ClassicControl), iter(CLASSIC_CONTROL)),
    ids=operator.itemgetter(1),
    scope="module",
)
def env(request) -> ClassicControl:
    env = request.param[0]()
    yield env
    env.close()


class TestClassic(TestPlangymEnv):
    def test_wrap_environment(self, env):
        if env.name == "Acrobot-v1":
            return None
        return super().test_wrap_environment(env)
