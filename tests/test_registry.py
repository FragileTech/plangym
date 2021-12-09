import pytest

from plangym.registry import make


atari = dict(name="MsPacman-ram-v0", clone_seeds=True, obs_type="ram")
classic_parallel = dict(name="CartPole-v0", n_workers=2)
retro_ray = dict(
    name="SonicTheHedgehog-Genesis",
    state="GreenHillZone.Act3",
    n_workers=2,
    ray=True,
    delay_init=True,
)
lunar_lander = dict(name="FastLunarLander-v0", frameskip=2)
bipedal_walker = dict(name="BipedalWalker-v3", frameskip=2)
minimal_pacman = dict(name="MinimalPacman-v0")
minimal_pong = dict(name="MinimalPong-v0")
plan_montexuma = dict(name="PlanMontezuma-v0")
dm_control = dict(task_name="walk", domain_name="walker")

all_envs = [
    atari,
    classic_parallel,
    retro_ray,
    lunar_lander,
    bipedal_walker,
    minimal_pacman,
    dm_control,
    minimal_pong,
    plan_montexuma,
]


@pytest.mark.parametrize("params", all_envs)
def test_make(params):
    env = make(**params)
