from collections import deque

from gym import ObservationWrapper
from gym.spaces import Box
from gym.wrappers import LazyFrames
import numpy


class FrameStack(ObservationWrapper):
    """
    Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation \
    contains the most recent 4 observations. For environment 'Pendulum-v0', \
    the original observation is an array with shape [3], so if we stack 4 \
    observations, the processed observation has shape [3, 4].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict` \
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): If ``True`` compress the frames to save even more memory.

    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = numpy.repeat(self.observation_space.low[numpy.newaxis, ...], num_stack, axis=0)
        high = numpy.repeat(self.observation_space.high[numpy.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()
