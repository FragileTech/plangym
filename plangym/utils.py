"""Generic utilities for working with environments."""
import gym


def has_time_limit(gym_env: gym.Env) -> bool:
    """Return True if the environment has a TimeLimit wrapper."""
    return hasattr(gym_env, "_max_episode_steps") and isinstance(
        gym_env,
        gym.wrappers.time_limit.TimeLimit,
    )


def remove_time_limit_from_spec(spec):
    """Remove the maximum time limit of an environment spec."""
    if hasattr(spec, "max_episode_steps"):
        spec._max_episode_steps = spec.max_episode_steps
    if hasattr(spec, "max_episode_time"):
        spec._max_episode_time = spec.max_episode_time
    spec.max_episode_steps = 1e100
    spec.max_episode_time = 1e100


def remove_time_limit(gym_env: gym.Env) -> gym.Env:
    """Remove the maximum time limit of the provided environment."""
    if has_time_limit(gym_env):
        gym_env._max_episode_steps = 1e100
        if hasattr(gym_env, "_max_episode_time"):
            gym_env._max_episode_time = 1e100
        if hasattr(gym_env, "spec"):
            remove_time_limit_from_spec(gym_env.spec)
    return gym_env
