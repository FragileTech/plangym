"""Generic utilities for working with environments."""

import os

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit
import numpy
from pyvirtualdisplay import Display
import cv2

try:
    from PIL import Image

    USE_PIL = True
except ImportError:  # pragma: no cover
    USE_PIL = False


def get_display(visible=False, size=(400, 400), **kwargs):
    """Start a virtual display."""
    os.environ["PYVIRTUALDISPLAY_DISPLAYFD"] = "0"
    display = Display(visible=visible, size=size, **kwargs)
    display.start()
    return display


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
            if isinstance(gym_env.env, gym.Wrapper) and isinstance(gym_env.env, TimeLimit):
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


def process_frame_pil(
    frame: numpy.ndarray,
    width: int | None = None,
    height: int | None = None,
    mode: str = "RGB",
) -> numpy.ndarray:
    """Resize an RGB frame to a specified shape and mode.

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
    mode = "L" if mode == "GRAY" else mode
    height = height or frame.shape[0]
    width = width or frame.shape[1]
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize(size=(width, height))
    return numpy.array(frame)


def process_frame_opencv(
    frame: numpy.ndarray,
    width: int | None = None,
    height: int | None = None,
    mode: str = "RGB",
) -> numpy.ndarray:
    """Resize an RGB frame to a specified shape and mode.

    Use OpenCV to resize an RGB frame to a specified height and width \
    or changing it to a different mode.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to cv2.cvtColor.

    Returns:
        The resized frame that matches the provided width and height.

    """
    height = height or frame.shape[0]
    width = width or frame.shape[1]
    frame = cv2.resize(frame, (width, height))
    if mode in {"GRAY", "L"}:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif mode == "BGR":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def process_frame(
    frame: numpy.ndarray,
    width: int | None = None,
    height: int | None = None,
    mode: str = "RGB",
) -> numpy.ndarray:
    """Resize an RGB frame to a specified shape and mode.

    Use either PIL or OpenCV to resize an RGB frame to a specified height and width \
    or changing it to a different mode.

    Args:
        frame: Target numpy array representing the image that will be resized.
        width: Width of the resized image.
        height: Height of the resized image.
        mode: Passed to either Image.convert or cv2.cvtColor.

    Returns:
        The resized frame that matches the provided width and height.

    """
    func = process_frame_pil if USE_PIL else process_frame_opencv  # pragma: no cover
    return func(frame, width, height, mode)


class GrayScaleObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Convert the image observation from RGB to gray scale.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import GrayscaleObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = GrayscaleObservation(gym.make("CarRacing-v3"))
        >>> env.observation_space
        Box(0, 255, (96, 96), uint8)
        >>> env = GrayscaleObservation(gym.make("CarRacing-v3"), keep_dim=True)
        >>> env.observation_space
        Box(0, 255, (96, 96, 1), uint8)

    """

    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. \
                observations are of the shape AxBx1. Otherwise, they are of shape AxB.

        """
        gym.utils.RecordConstructorArgs.__init__(self, keep_dim=keep_dim)
        gym.ObservationWrapper.__init__(self, env)

        self.keep_dim = keep_dim

        assert (
            "Box" in self.observation_space.__class__.__name__  # works for both gym and gymnasium
            and len(self.observation_space.shape) == 3  # noqa: PLR2004
            and self.observation_space.shape[-1] == 3  # noqa: PLR2004
        ), f"Expected input to be of shape (..., 3), got {self.observation_space.shape}"

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=numpy.uint8
            )
        else:
            self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=numpy.uint8)

    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment."""
        # Avoid infinite recursion by checking if 'env' exists
        if name == "env":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'env'")
        return getattr(self.env, name)

    def observation(self, observation):
        """Convert the colour observation to greyscale.

        Args:
            observation: Color observations

        Returns:
            Grayscale observations

        """
        import cv2  # noqa: PLC0415

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = numpy.expand_dims(observation, -1)
        return observation
