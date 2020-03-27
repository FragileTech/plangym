from gym import ObservationWrapper
import gym.spaces as spaces
import numpy


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        flatdim = spaces.flatdim(env.observation_space)
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(flatdim,), dtype=numpy.float32
        )

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)
