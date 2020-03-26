"""
Environments and wrappers for Sonic training.
"""
from collections import deque

import gym
from gym import Env, error, spaces
import numpy as np

from plangym import AtariEnvironment, BaseEnvironment


class Wrapper(gym.Env):
    env = None

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self._warn_double_wrap()

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __getattr__(self, item):
        return getattr(self.env, item)

    def _warn_double_wrap(self):
        env = self.env
        while True:
            if isinstance(env, Wrapper):
                if env.class_name() == self.class_name():
                    raise error.DoubleWrapperError(
                        "Attempted to double wrap with Wrapper: "
                        "{}".format(self.__class__.__name__)
                    )
                env = env.env
            else:
                break

    def step(self, action, *args, **kwargs):
        if isinstance(self.env, BaseEnvironment):
            return self.env.step(action, *args, **kwargs)
        else:
            return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        if self.env:
            return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec


class ActionWrapper(Wrapper):
    def step(self, action, *args, **kwargs):
        action = self.action(action)
        if isinstance(self.env, BaseEnvironment):
            return self.env.step(action, *args, **kwargs)
        else:
            return self.env.step(action)

    def reverse_action(self, action):
        return self._reverse_action(action)


class AtariWrapper(AtariEnvironment):

    """Generic wrapper for the AtariEnvironment. Inherit from this class and override
     the desired methods to implement a wrapper."""

    def __init__(
        self,
        env: Env,
        clone_seeds: bool = True,
        n_repeat_action: int = 1,
        min_dt: int = 1,
        obs_ram: bool = False,
        episodic_live: bool = False,
        autoreset: bool = True,
    ):
        """Create a wrapper for the AtariEnvironment.
        :param clone_seeds: Clone the random seed of the ALE emulator when reading/setting
         the state.
        :param n_repeat_action: Consecutive number of times a given action will be applied.
        :param min_dt: Internal number of times an action will be applied for each step in
         n_repeat_action.
        :param obs_ram: Use ram as observations even though it is not specified elsewhere.
        :param episodic_live: Return end = True when losing a live.
        :param autoreset: Restart environment when reaching a terminal state.
        :param env: Instance of AtariEnvironment, or a wrapped AtariEnvironment.
        """
        super(AtariWrapper, self).__init__(
            name=env.spec.id,
            clone_seeds=clone_seeds,
            n_repeat_action=n_repeat_action,
            min_dt=min_dt,
            obs_ram=obs_ram,
            episodic_live=episodic_live,
            autoreset=autoreset,
        )
        # Overwrite the default env with the one provided externally
        self._env = env
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata


class SonicDiscretizer(ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [
            ["LEFT"],
            ["RIGHT"],
            ["LEFT", "DOWN"],
            ["RIGHT", "DOWN"],
            ["DOWN"],
            ["DOWN", "B"],
            ["B"],
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        return reward * 0.01


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
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


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8
        )

    def __getattr__(self, item):
        return getattr(self.env, item)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
