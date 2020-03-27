from collections import deque

from gym import ObservationWrapper
from gym.spaces import Box
import numpy


class LazyFrames:
    def __init__(self, frames, lz4_compress: bool = False):
        """
        This object ensures that common frames between the observations are only stored once.

        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        To further reduce the memory use, it is optionally to turn on lz4 to \
        compress the observations.
        """
        if lz4_compress:
            from lz4.block import compress

            self.shape = frames[0].shape
            self.dtype = frames[0].dtype
            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self._out = None
        self.lz4_compress = lz4_compress

    def _force(self):
        if self._out is None:
            if self.lz4_compress:
                from lz4.block import decompress

                frames = [
                    numpy.frombuffer(decompress(frame), dtype=self.dtype).reshape(self.shape)
                    for frame in self._frames
                ]
            else:
                frames = self._frames
            self._out = numpy.concatenate(frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


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
