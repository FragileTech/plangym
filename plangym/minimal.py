import copy

from gym import spaces
import numpy as np

from plangym.atari import AtariEnvironment
from plangym.utils import resize_frame


class MinimalPong(AtariEnvironment):
    """
    Environment adapted to play pong returning the smallest observation possible.

    This is meant for testing RL algorithms. The number of possible actions \
    has been reduced to 2, and it returns an observation that is 80x80x1 pixels.
    """

    def __init__(self, name="Pong-v4", *args, **kwargs):
        """Initialize a :class:`MinimalPong`."""
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
        return obs.astype(np.float)

    def step(self, action: np.ndarray, state: np.ndarray = None, dt: int = None) -> tuple:
        if state is not None:
            self.set_state(state)
        final_obs = np.zeros((80, 80, 2))
        action = 2 if action == 0 else 3
        reward = 0
        end, _end = False, False
        info = {"lives": -1}
        for i in range(2):
            obs, _reward, _end, _info = self.gym_env.step(action)
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
            self.gym_env.reset()
        return data

    def reset(self, return_state: bool = True):
        obs = self.gym_env.reset()
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

    def __init__(self, name: str = "MsPacman-v4", *args, **kwargs):
        obs_shape = kwargs.get("obs_shape", (80, 80, 2))
        # Do not pas obs_shape to AtariEnvironment
        if "obs_shape" in kwargs.keys():
            del kwargs["obs_shape"]
        super(MinimalPacman, self).__init__(name=name, *args, **kwargs)
        self.obs_shape = obs_shape
        # Im freezing this until proven wrong
        self.min_dt = 4
        self.dt = 1
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

    def step(self, action: np.ndarray, state: np.ndarray = None, dt: int = None) -> tuple:
        dt = dt if dt is not None else self.dt
        if state is not None:
            self.set_state(state)
        reward = 0
        end, _end = False, False
        info = {"lives": -1, "reward": 0}
        for _ in range(dt):
            full_obs = np.zeros(self.observation_space.shape)
            obs_hist = []
            for _ in range(self.min_dt):
                obs, _reward, _end, _info = self.gym_env.step(action)
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
                full_obs[:, :, 0] = obs_hist[-1][:, :, 0]
            if len(obs_hist) > 1:
                filtered = self.normalize_vector(np.array(obs_hist))
                full_obs[:, :, 1] = filtered[-1][:, :, 0]

            if _end:
                break
        info["terminal"] = _end
        if state is not None:
            new_state = self.get_state()
            return new_state, full_obs, reward, end, info
        return full_obs, reward, end, info

    def reset(self, return_state: bool = True):
        full_obs = np.zeros(self.observation_space.shape)
        obs = self.reshape_frame(self.gym_env.reset())
        obs_hist = [copy.deepcopy(obs)]
        reward = 0
        end = False
        info = {"lives": -1}
        for _ in range(3):

            obs, _reward, _end, _info = self.gym_env.step(0)
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
            full_obs[:, :, 0] = obs_hist[-1][:, :, 0]
        if len(obs_hist) > 1:
            filtered = self.normalize_vector(np.array(obs_hist))
            full_obs[:, :, 1] = filtered[-1][:, :, 0]
        if not return_state:
            return full_obs
        else:

            return self.get_state(), full_obs
