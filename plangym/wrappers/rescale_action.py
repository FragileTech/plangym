import gym
from gym import spaces
import numpy


class RescaleAction(gym.ActionWrapper):
    r"""Rescales the continuous action space of the environment to a range [a,b].

    Example::

        >>> RescaleAction(env, a, b).action_space == Box(a,b)
        True

    """

    def __init__(self, env, a, b):
        assert isinstance(
            env.action_space,
            spaces.Box,
        ), "expected Box action space, got {}".format(type(env.action_space))
        assert numpy.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = numpy.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = numpy.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = spaces.Box(
            low=a,
            high=b,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        assert numpy.all(numpy.greater_equal(action, self.a)), (action, self.a)
        assert numpy.all(numpy.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = numpy.clip(action, low, high)
        return action
