"""Functionality for instantiating the environment by passing the environment id."""
from plangym.control import BalloonEnv, Box2DEnv, ClassicControl, DMControlEnv, LunarLander
from plangym.environment_names import ATARI, BOX_2D, CLASSIC_CONTROL, DM_CONTROL, RETRO
from plangym.vectorization import ParallelEnv, RayEnv
from plangym.videogames import AtariEnv, MarioEnv, MontezumaEnv, RetroEnv


def get_planenv_class(name, domain_name, state):
    """Return the class corresponding to the environment name."""
    # if name == "MinimalPacman-v0":
    #    return MinimalPacman
    # elif name == "MinimalPong-v0":
    #    return MinimalPong
    if name == "PlanMontezuma-v0":
        return MontezumaEnv
    elif state is not None or name in set(RETRO):
        return RetroEnv
    elif name in set(CLASSIC_CONTROL):
        return ClassicControl
    elif name in set(BOX_2D):
        if name == "FastLunarLander-v0":
            return LunarLander
        return Box2DEnv
    elif name in ATARI:
        return AtariEnv
    elif domain_name is not None or any(x[0] in name for x in DM_CONTROL):
        return DMControlEnv
    elif "SuperMarioBros" in name:
        return MarioEnv
    elif "BalloonLearningEnvironment-v0":
        return BalloonEnv
    raise ValueError(f"Environment {name} is not supported.")


def get_environment_class(
    name: str = None,
    n_workers: int = None,
    ray: bool = False,
    domain_name: str = None,
    state: str = None,
):
    """Get the class and vectorized environment and PlangymEnv class from the make params."""
    env_class = get_planenv_class(name, domain_name, state)
    if ray:
        return RayEnv, env_class
    elif n_workers is not None:
        return ParallelEnv, env_class
    return None, env_class


def make(
    name: str = None,
    n_workers: int = None,
    ray: bool = False,
    domain_name: str = None,
    state: str = None,
    **kwargs,
):
    """Create the appropriate PlangymEnv from the environment name and other parameters."""
    parallel_class, env_class = get_environment_class(
        name=name,
        n_workers=n_workers,
        ray=ray,
        domain_name=domain_name,
        state=state,
    )
    kwargs["name"] = name
    if state is not None:
        kwargs["state"] = state
    if domain_name is not None:
        kwargs["domain_name"] = domain_name
    if parallel_class is not None:
        return parallel_class(env_class=env_class, n_workers=n_workers, **kwargs)
    return env_class(**kwargs)
