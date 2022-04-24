"""Generic utilities for working with environments."""
import gym
from gym.wrappers.time_limit import TimeLimit
import numpy
from PIL import Image


def remove_time_limit_from_spec(spec):
    """Remove the maximum time limit of an environment spec."""
    if hasattr(spec, "max_episode_steps"):
        spec._max_episode_steps = spec.max_episode_steps
        spec.max_episode_steps = 1e100
    if hasattr(spec, "max_episode_time"):
        spec._max_episode_time = spec.max_episode_time
        spec.max_episode_time = 1e100


def remove_time_limit(gym_env: gym.Env) -> gym.Env:
    """Remove the maximum time limit of the provided environment."""
    if hasattr(gym_env, "spec") and gym_env.spec is not None:
        remove_time_limit_from_spec(gym_env.spec)
    if not isinstance(gym_env, gym.Wrapper):
        return gym_env
    for _ in range(5):
        try:
            if isinstance(gym_env, TimeLimit):
                return gym_env.env
            elif isinstance(gym_env.env, gym.Wrapper) and isinstance(gym_env.env, TimeLimit):
                gym_env.env = gym_env.env.env
            # This is an ugly hack to make sure that we can remove the TimeLimit even
            # if somebody is crazy enough to apply three other wrappers on top of the TimeLimit
            elif isinstance(gym_env.env.env, gym.Wrapper) and isinstance(
                gym_env.env.env,
                TimeLimit,
            ):  # pragma: no cover
                gym_env.env.env = gym_env.env.env.env
            elif isinstance(gym_env.env.env.env, gym.Wrapper) and isinstance(
                gym_env.env.env.env,
                TimeLimit,
            ):  # pragma: no cover
                gym_env.env.env.env = gym_env.env.env.env.env
            else:  # pragma: no cover
                break
        except AttributeError:
            break
    return gym_env


def process_frame(
    frame: numpy.ndarray,
    width: int = None,
    height: int = None,
    mode: str = "RGB",
) -> numpy.ndarray:
    """
    Use PIL to resize an RGB frame to a specified height and width \
    or changing it to a different mode.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    height = height or frame.shape[0]
    width = width or frame.shape[1]
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize(size=(width, height))
    return numpy.array(frame)
