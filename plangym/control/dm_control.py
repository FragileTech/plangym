"""Implement the ``plangym`` API for ``dm_control`` environments."""
from typing import Iterable, Optional
import warnings

from gym.spaces import Box
import numpy as np

from plangym.core import PlangymEnv, wrap_callable


try:
    from gym.envs.classic_control import rendering

    novideo_mode = False
except Exception:  # pragma: no cover
    novideo_mode = True


class DMControlEnv(PlangymEnv):
    """
    Wrap the dm_control library, so it can work for planning problems.

    It allows parallel and vectorized execution of the environments.
    """

    DEFAULT_OBS_TYPE = "coords"

    def __init__(
        self,
        name: str = "cartpole-balance",
        frameskip: int = 1,
        episodic_life: bool = False,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_setup: bool = False,
        visualize_reward: bool = True,
        domain_name=None,
        task_name=None,
        render_mode=None,
        obs_type: Optional[str] = None,
        remove_time_limit=None,
    ):
        """
        Initialize a :class:`DMControlEnv`.

        Args:
            name: Provide the task to be solved as `domain_name-task_name`. For
                  example 'cartpole-balance'.
            frameskip: Set a deterministic frameskip to apply the same
                       action N times.
            episodic_life: Send terminal signal after loosing a life.
            autoreset: Restart environment when reaching a terminal state.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            delay_setup: If ``True`` do not initialize the ``gym.Environment`` \
                      and wait for ``setup`` to be called later.
            visualize_reward: The color of the agent depends on the reward on it's last timestep.
            domain_name: Same as in dm_control.suite.load.
            task_name: Same as in dm_control.suite.load.
            render_mode: None|human|rgb_array
        """
        self._visualize_reward = visualize_reward
        self.viewer = []
        self._viewer = None
        name, self._domain_name, self._task_name = self._parse_names(name, domain_name, task_name)
        super(DMControlEnv, self).__init__(
            name=name,
            frameskip=frameskip,
            episodic_life=episodic_life,
            wrappers=wrappers,
            delay_setup=delay_setup,
            autoreset=autoreset,
            render_mode=render_mode,
            obs_type=obs_type,
        )

    @property
    def physics(self):
        """Alias for gym_env.physics."""
        return self.gym_env.physics

    @property
    def domain_name(self) -> str:
        """Return the name of the agent in the current simulation."""
        return self._domain_name

    @property
    def task_name(self) -> str:
        """Return the name of the task in the current simulation."""
        return self._task_name

    @staticmethod
    def _parse_names(name, domain_name, task_name):
        if isinstance(name, str) and domain_name is None:
            domain_name = name if "-" not in name else name.split("-")[0]

        if isinstance(name, str) and "-" in name and task_name is None:
            task_name = task_name if "-" not in name else name.split("-")[1]
        if (not isinstance(name, str) or "-" not in name) and task_name is None:
            raise ValueError(
                f"Invalid combination: name {name},"
                f" domain_name {domain_name}, task_name {task_name}",
            )
        name = "-".join([domain_name, task_name])
        return name, domain_name, task_name

    def init_gym_env(self):
        """Initialize the environment instance that the current class is wrapping."""
        from dm_control import suite

        env = suite.load(
            domain_name=self.domain_name,
            task_name=self.task_name,
            visualize_reward=self._visualize_reward,
        )
        self.viewer = []
        self._viewer = None if novideo_mode else rendering.SimpleImageViewer()
        return env

    def setup(self):
        """Initialize the target :class:`gym.Env` instance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(DMControlEnv, self).setup()

    def _init_action_space(self):
        self._action_space = Box(
            low=self.action_spec().minimum,
            high=self.action_spec().maximum,
            dtype=np.float32,
        )

    def _init_obs_space_coords(self):
        shape = self.reset(return_state=False).shape
        self._obs_space = Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def action_spec(self):
        """Alias for the environment's ``action_spec``."""
        return self.gym_env.action_spec()

    def get_image(self) -> np.ndarray:
        """
        Return a numpy array containing the rendered view of the environment.

        Square matrices are interpreted as a greyscale image. Three-dimensional arrays
        are interpreted as RGB images with channels (Height, Width, RGB)
        """
        return self.gym_env.physics.render(camera_id=0)

    def render(self, mode="human"):
        """
        Store all the RGB images rendered to be shown when the `show_game`\
        function is called.

        Args:
            mode: `rgb_array` return an RGB image stored in a numpy array. `human`
             stores the rendered image in a viewer to be shown when `show_game`
             is called.

        Returns:
            numpy.ndarray when mode == `rgb_array`. True when mode == `human`
        """
        img = self.get_image()
        if mode == "rgb_array":
            return img
        elif mode == "human":
            self.viewer.append(img)
        return True

    def show_game(self, sleep: float = 0.05):
        """Render the collected RGB images."""
        import time

        for img in self.viewer:
            self._viewer.imshow(img)
            time.sleep(sleep)

    def get_coords_obs(self, obs, **kwargs) -> np.ndarray:
        """
        Get the environment observation from a time_step object.

        Args:
            obs: Time step object returned after stepping the environment.
            **kwargs: Ignored

        Returns:
            Numpy array containing the environment observation.
        """
        return self._time_step_to_obs(time_step=obs)

    def set_state(self, state: np.ndarray) -> None:
        """
        Set the state of the simulator to the target State.

        Args:
            state: numpy.ndarray containing the information about the state to be set.

        Returns:
            None
        """
        with self.gym_env.physics.reset_context():
            self.gym_env.physics.set_state(state)

    def get_state(self) -> np.ndarray:
        """
        Return a tuple containing the three arrays that characterize the state\
         of the system.

        Each tuple contains the position of the robot, its velocity
         and the control variables currently being applied.

        Returns:
            Tuple of numpy arrays containing all the information needed to describe
            the current state of the simulation.
        """
        return self.gym_env.physics.get_state()

    def apply_action(self, action):
        """Transform the returned time_step object to a compatible gym tuple."""
        info = {}
        time_step = self.gym_env.step(action)
        obs = time_step
        terminal = time_step.last()
        _reward = time_step.reward if time_step.reward is not None else 0.0
        reward = _reward + self._reward_step
        return obs, reward, terminal, info

    @staticmethod
    def _time_step_to_obs(time_step) -> np.ndarray:
        # Concat observations in a single array, so it is easier to calculate distances
        obs_array = np.hstack(
            [np.array([time_step.observation[x]]).flatten() for x in time_step.observation],
        )
        return obs_array

    def close(self):
        """Tear down the environment and close rendering."""
        try:
            super(DMControlEnv, self).close()
            if self._viewer is not None:
                self._viewer.close()
        except Exception:
            pass
