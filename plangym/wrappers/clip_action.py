from gym import ActionWrapper
from gym.spaces import Box
import numpy


class ClipAction(ActionWrapper):
    r"""Clip the continuous action within the valid bound."""

    def __init__(self, env):
        assert isinstance(env.action_space, Box)
        super(ClipAction, self).__init__(env)

    def action(self, action):
        return numpy.clip(action, self.action_space.low, self.action_space.high)
