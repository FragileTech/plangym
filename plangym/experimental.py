from typing import Callable

import gym
import numpy as np

from plangym.env import Environment


class ESEnvironment(Environment):
    """DO NOT USE: NOT FINISHED!! Environment for Solving Evolutionary Strategies."""

    def __init__(
        self,
        name: str,
        dnn_callable: Callable,
        n_repeat_action: int = 1,
        max_episode_length=1000,
        noise_prob: float = 0,
    ):
        super(ESEnvironment, self).__init__(name=name, n_repeat_action=n_repeat_action)
        self.dnn_callable = dnn_callable
        self._env = gym.make(name)
        self.neural_network = self.dnn_callable()
        self.max_episode_length = max_episode_length
        self.noise_prob = noise_prob

    def __getattr__(self, item):
        return getattr(self._env, item)

    def get_state(self) -> np.ndarray:
        return self.neural_network.get_weights()

    def set_state(self, state: [np.ndarray, list]):
        """
        Sets the microstate of the simulator to the microstate of the target State.
        I will be super grateful if someone shows me how to do this using Open Source code.
        :param state:
        :return:
        """
        self.neural_network.set_weights(state)

    @staticmethod
    def _perturb_weights(weights: [list, np.ndarray], perturbations: [list, np.ndarray]) -> list:
        """
        Updates a set of weights with a gaussian perturbation with sigma equal to self.sigma and
        mean 0.
        :param weights: Set of weights that will be updated.
        :param perturbations: Standard gaussian noise.
        :return: perturbed weights with desired sigma.
        """
        weights_try = []
        for index, noise in enumerate(perturbations):
            weights_try.append(weights[index] + noise)
        return weights_try

    def _normalize_observation(self, obs):
        if "v0" in self.name:
            return obs / 255
        else:
            return obs

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None
    ) -> tuple:

        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action

        if state is not None:
            new_weights = self._perturb_weights(state, action)
            self.set_state(new_weights)
        obs = self._env.reset()
        reward = 0
        n_steps = 0
        end = False
        while not end and n_steps < self.max_episode_length:
            if np.random.random() < self.noise_prob:
                nn_action = self._env.action_space.sample()
            else:
                processed_obs = self._normalize_observation(obs.flatten())
                nn_action = self.neural_network.predict(processed_obs)
            for _ in range(n_repeat_action):

                obs, _reward, end, info = self._env.step(nn_action)
                reward += _reward
                n_steps += 1

        if state is not None:
            new_state = self.get_state()
            return new_state, obs, reward, False, 0
        return obs, reward, False, 0

    def step_batch(self, actions, states=None, n_repeat_action: int = None) -> tuple:
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        n_repeat_action = (
            n_repeat_action
            if isinstance(n_repeat_action, np.ndarray)
            else np.ones(len(states)) * n_repeat_action
        )
        data = [
            self.step(action, state, n_repeat_action=dt)
            for action, state, dt in zip(actions, states, n_repeat_action)
        ]
        new_states, observs, rewards, terminals, lives = [], [], [], [], []
        for d in data:
            if states is None:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            lives.append(info)
        if states is None:
            return observs, rewards, terminals, lives
        else:
            return new_states, observs, rewards, terminals, lives

    def reset(self, return_state: bool = False):
        if not return_state:
            return self._env.reset()
        else:
            obs = self._env.reset()
            return self.get_state(), obs
