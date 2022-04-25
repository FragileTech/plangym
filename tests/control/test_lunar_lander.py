import pytest


pytest.importorskip("Box2D")
from plangym.api_tests import (  # noqa: F401
    batch_size,
    display,
    generate_test_cases,
    TestPlanEnv,
    TestPlangymEnv,
)
from plangym.control.lunar_lander import FastGymLunarLander, LunarLander


def lunar_lander_det_discrete():
    return LunarLander(autoreset=False, deterministic=True, continuous=False)


def lunar_lander_random_discrete():
    return LunarLander(autoreset=False, deterministic=False, continuous=False)


def lunar_lander_random_continuous():
    return LunarLander(
        autoreset=False,
        deterministic=False,
        continuous=True,
    )


environments = [
    lunar_lander_det_discrete,
    lunar_lander_random_discrete,
    lunar_lander_random_continuous,
]


@pytest.fixture(
    params=generate_test_cases(["FastLunarLander-v0"], LunarLander, custom_tests=environments),
    scope="module",
)
def env(request) -> LunarLander:
    env = request.param()
    yield env
    env.close()


class TestFastGymLunarLander:
    def test_death(self):
        gym_env = FastGymLunarLander()
        gym_env.reset()
        for _ in range(1000):
            *_, end, info = gym_env.step(gym_env.action_space.sample())
            if end:
                break
