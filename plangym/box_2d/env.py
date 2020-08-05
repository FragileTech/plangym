import copy
from typing import Union
import pickle

import numpy

from Box2D.Box2D import b2Vec2, b2Transform
from plangym.core import GymEnvironment

import copy


class Box2DEnv(GymEnvironment):
    def get_state(self) -> numpy.array:
        """
        Recover the internal state of the simulation.

        An state must completely describe the Environment at a given moment.
        """
        state = get_env_state(self.gym_env)  # pickle.dumps(get_env_state(self.gym_env))

        # state_vector = numpy.zeros(200, dtype=object)
        # state_vector[: len(state)] = tuple(state[:])[:]
        # if len(state.shape) == 1:
        #    state = state[numpy.newaxis, :]
        return numpy.array((state, None), dtype=object)  # "S250000")

    def set_state(self, state: numpy.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        # loaded_state = pickle.loads(state[:])
        set_env_state(self.gym_env, state[0])

    def _lunar_lander_end(self, obs):
        if self.gym_env.game_over or abs(obs[0]) >= 1.0:
            return True
        elif not self.gym_env.lander.awake:
            return True
        return False

    def _step_with_dt(self, action, dt):
        obs, reward, _, info = super(Box2DEnv, self)._step_with_dt(action, dt)
        terminal = self._lunar_lander_end(obs)
        info["oob"] = terminal
        return obs, reward, terminal, info
