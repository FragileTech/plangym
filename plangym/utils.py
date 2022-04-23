"""Generic utilities for working with environments."""
from typing import Union

import cv2
import gym
from gym.wrappers.time_limit import TimeLimit
import numpy
from PIL import Image


cv2.ocl.setUseOpenCL(False)


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
            elif isinstance(gym_env.env.env, gym.Wrapper) and isinstance(
                gym_env.env.env,
                TimeLimit,
            ):
                gym_env.env.env = gym_env.env.env.env
            elif isinstance(gym_env.env.env.env, gym.Wrapper) and isinstance(
                gym_env.env.env.env,
                TimeLimit,
            ):
                gym_env.env.env.env = gym_env.env.env.env.env
            else:
                break
        except AttributeError:
            break
    return gym_env


def resize_frame(
    frame: numpy.ndarray,
    width: int,
    height: int,
    mode: str = "RGB",
) -> numpy.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize(size=(width, height))
    return numpy.array(frame)


class Rgb2gray(gym.ObservationWrapper):
    """Transform RGB images to greyscale."""

    def __init__(self, env):
        """Transform RGB images to greyscale."""
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, _oldc) = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(oldh, oldw, 1),
            dtype=numpy.uint8,
        )

    def observation(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Return observation as a greyscale image."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame[:, :, None]


class Downsample(gym.ObservationWrapper):
    """Downsample observation by a factor of ratio."""

    def __init__(self, env: gym.Env, ratio: Union[int, float]):
        """Downsample images by a factor of ratio."""
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (oldh // ratio, oldw // ratio, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=newshape, dtype=numpy.uint8)

    def observation(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Return the downsampled observation."""
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        return frame
