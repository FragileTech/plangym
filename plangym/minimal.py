import copy

import gym
from gym import spaces
import numpy as np


from plangym.env import AtariEnvironment, Environment, resize_frame


class ClassicControl(Environment):
    """Environment for playing Atari games."""

    def __init__(self, name: str = "CartPole-v1"):
        super(ClassicControl, self).__init__(name=name, n_repeat_action=1)
        self._env = gym.make(name)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata
        self.min_dt = 1

    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def n_actions(self):
        return self._env.action_space.n

    def get_state(self) -> np.ndarray:
        return np.array(copy.copy(self._env.unwrapped.state))

    def set_state(self, state: np.ndarray):
        self._env.unwrapped.state = copy.copy(tuple(state.tolist()))
        return state

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = 1
    ) -> tuple:
        if state is not None:
            self.set_state(state)
        info = {"lives": 1}

        obs, reward, end, _info = self._env.step(action)
        info["terminal"] = end
        if state is not None:
            new_state = self.get_state()
            data = new_state, obs, reward, end, info
        else:
            data = obs, reward, end, info
        if end:
            self._env.reset()
        return data

    def step_batch(self, actions, states=None, n_repeat_action: [int, np.ndarray] = None) -> tuple:
        """

        :param actions:
        :param states:
        :param n_repeat_action:
        :return:
        """
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

    def reset(self, return_state: bool = True):
        if not return_state:
            return self._env.reset()
        else:
            obs = self._env.reset()
            return self.get_state(), obs

    def render(self):
        return self._env.render()


class MinimalPong(AtariEnvironment):
    """Minimal pong environment"""

    def __init__(self, name="PongNoFrameskip-V4", *args, **kwargs):
        """Environment adapted to play pong returning the smallest observation possible.
        This is meant for testing RL algos. The number of possible actions has been reduced to 2,
         and it returns an observation that is 80x80x1 pixels."""
        super(MinimalPong, self).__init__(name=name, *args, **kwargs)
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.float, shape=(80, 80))
        self.action_space = spaces.Discrete(2)

    @staticmethod
    def process_obs(obs):
        """Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
        This was copied from Andrej Karpathy's blog."""
        obs = obs[35:195]  # Crop
        obs = obs[::2, ::2, 0]  # Downsample by factor of 2
        obs[obs == 144] = 0  # Erase background (background type 1)
        obs[obs == 109] = 0  # Erase background (background type 2)
        obs[obs != 0] = 1  # Everything else (paddles, ball) just set to 1
        return obs.astype(np.float)  # .ravel()

    @property
    def n_actions(self):
        return 2

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None
    ) -> tuple:
        if state is not None:
            self.set_state(state)
        final_obs = np.zeros((80, 80, 2))
        action = 2 if action == 0 else 3
        reward = 0
        end, _end = False, False
        info = {"lives": -1}
        for i in range(2):
            obs, _reward, _end, _info = self._env.step(action)
            if "ram" not in self.name:
                cur_x = self.process_obs(obs)
                final_obs[:, :, i] = cur_x.copy()
            else:
                final_obs = obs
            _info["lives"] = _info.get("ale.lives", -1)
            end = _end or end or info["lives"] > _info["lives"]
            info = _info.copy()
            reward += _reward
            if _end:
                break
        info["terminal"] = _end
        if state is not None:
            new_state = self.get_state()
            data = new_state, final_obs, reward, end, info
        else:
            data = final_obs, reward, end, info
        if _end:
            self._env.reset()
        return data

    def reset(self, return_state: bool = True):
        obs = self._env.reset()
        if "ram" not in self.name:
            proc_obs = np.zeros((80, 80, 2))
            proc_obs[:, :, 0] = self.process_obs(obs)
        else:
            proc_obs = obs
        if not return_state:
            return proc_obs
        else:
            return self.get_state(), proc_obs


class MinimalPacman(AtariEnvironment):
    """Minimal pacman environment"""

    def __init__(self, *args, **kwargs):
        obs_shape = kwargs.get("obs_shape", (80, 80, 2))
        # Do not pas obs_shape to AtariEnvironment
        if "obs_shape" in kwargs.keys():
            del kwargs["obs_shape"]
        super(MinimalPacman, self).__init__(*args, **kwargs)
        self.obs_shape = obs_shape
        # Im freezing this until proven wrong
        self.min_dt = 4
        self.n_repeat_action = 1
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.float, shape=obs_shape)

    @staticmethod
    def normalize_vector(vector):
        std = vector.std(axis=0)
        std[std == 0] = 1

        standard = (vector - vector.mean(axis=0)) / np.minimum(1e-4, std)
        standard[standard > 0] = np.log(1 + standard[standard > 0]) + 1
        standard[standard <= 0] = np.exp(standard[standard <= 0])
        return standard

    def reshape_frame(self, obs):
        height, width = self.obs_shape[0], self.obs_shape[1]
        cropped = obs[3:170, 7:-7]
        frame = resize_frame(cropped, width=width, height=height)
        return frame

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None
    ) -> tuple:
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        if state is not None:
            self.set_state(state)
        reward = 0
        end, _end = False, False
        info = {"lives": -1, "reward": 0}
        for _ in range(n_repeat_action):
            full_obs = np.zeros(self.observation_space.shape)
            obs_hist = []
            for _ in range(self.min_dt):
                obs, _reward, _end, _info = self._env.step(action)
                _info["lives"] = _info.get("ale.lives", -1)
                _info["reward"] = float(info["reward"])

                end = _end or end or info["lives"] > _info["lives"]
                if end:
                    reward -= 1000

                info = _info.copy()
                info["reward"] += _reward
                reward += _reward
                if _end:
                    break
                proced = self.reshape_frame(obs)
                obs_hist.append(proced)

            if len(obs_hist) > 0:
                full_obs[:, :, 0] = obs_hist[-1]
            if len(obs_hist) > 1:
                filtered = self.normalize_vector(np.array(obs_hist))
                full_obs[:, :, 1] = filtered[-1]

            if _end:
                break
        info["terminal"] = _end
        if state is not None:
            new_state = self.get_state()
            return new_state, full_obs, reward, end, info
        return full_obs, reward, end, info

    def reset(self, return_state: bool = True):
        full_obs = np.zeros(self.observation_space.shape)
        obs = self.reshape_frame(self._env.reset())
        obs_hist = [copy.deepcopy(obs)]
        reward = 0
        end = False
        info = {"lives": -1}
        for _ in range(3):

            obs, _reward, _end, _info = self._env.step(0)
            _info["lives"] = _info.get("ale.lives", -1)
            end = _end or end or info["lives"] > _info["lives"]
            if end:
                reward -= 1000
            info = _info.copy()
            reward += _reward
            if _end:
                break
            proced = self.reshape_frame(obs)
            obs_hist.append(proced)

        if len(obs_hist) > 0:
            full_obs[:, :, 0] = obs_hist[-1]
        if len(obs_hist) > 1:
            filtered = self.normalize_vector(np.array(obs_hist))
            full_obs[:, :, 1] = filtered[-1]
        if not return_state:
            return full_obs
        else:

            return self.get_state(), full_obs
