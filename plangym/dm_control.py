"""Implement the ``plangym`` API for ``dm_control`` environments."""
from typing import Iterable, Union
import warnings

from gym.spaces import Box
import numpy as np

from plangym.core import PlanEnvironment, wrap_callable


try:
    from gym.envs.classic_control import rendering

    novideo_mode = False
except Exception:
    novideo_mode = True


class DMControlEnv(PlanEnvironment):
    """
    Wrap the dm_control library, so it can work for planning problems.

    It allows parallel and vectorized execution of the environments.
    """

    def __init__(
        self,
        name: str = "cartpole-balance",
        frameskip: int = 1,
        episodic_live: bool = False,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_init: bool = False,
        visualize_reward: bool = True,
        domain_name=None,
        task_name=None,
        render_mode=None,
    ):
        """
        Initialize a :class:`DMControlEnv`.

        Args:
            name: Provide the task to be solved as `domain_name-task_name`. For
                  example 'cartpole-balance'.
            frameskip: Set a deterministic frameskip to apply the same
                       action N times.
            episodic_live: Send terminal signal after loosing a life.
            autoreset: Restart environment when reaching a terminal state.
            wrappers: Wrappers that will be applied to the underlying OpenAI env. \
                     Every element of the iterable can be either a :class:`gym.Wrapper` \
                     or a tuple containing ``(gym.Wrapper, kwargs)``.
            delay_init: If ``True`` do not initialize the ``gym.Environment`` \
                      and wait for ``init_env`` to be called later.
            visualize_reward: The color of the agent depends on the reward on it's last timestep.
            domain_name: Same as in dm_control.suite.load.
            task_name: Same as in dm_control.suite.load.
            render_mode: None|human|rgb_array
        """
        self._visualize_reward = visualize_reward
        self._render_i = 0
        self.viewer = []
        self._last_time_step = None
        self._viewer = None
        self._render_mode = render_mode
        name, self._domain_name, self._task_name = self._parse_names(name, domain_name, task_name)
        super(DMControlEnv, self).__init__(
            name=name,
            frameskip=frameskip,
            episodic_live=episodic_live,
            wrappers=wrappers,
            delay_init=delay_init,
            autoreset=autoreset,
        )

    @property
    def physics(self):
        """Alias for gym_env.physics."""
        return self.gym_env.physics

    @property
    def render_mode(self) -> str:
        """Return how the game will be rendered. Values: None | human | rgb_array."""
        return self._render_mode

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
        self._render_i = 0
        self.viewer = []
        self._last_time_step = None
        self._viewer = None if novideo_mode else rendering.SimpleImageViewer()
        return env

    def init_env(self):
        """Initialize the target :class:`gym.Env` instance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gym_env = self.init_gym_env()
            if self._wrappers is not None:
                self.apply_wrappers(self._wrappers)
            shape = self.reset(return_state=False).shape
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
            self.action_space = Box(
                low=self.action_spec().minimum,
                high=self.action_spec().maximum,
                dtype=np.float32,
            )

    def action_spec(self):
        """Alias for the environment's ``action_spec``."""
        return self.gym_env.action_spec()

    def action_space(self):
        """Alias for the environment's ``action_spec``."""
        return self.gym_env.action_spec()

    def seed(self, seed=None):
        """Seed the underlying :class:`gym.Env`."""
        np.random.seed(seed)
        # self.gym_env.seed(seed)

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
        img = self.gym_env.physics.render(camera_id=0)
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

    def reset(self, return_state: bool = True) -> [np.ndarray, tuple]:
        """
        Reset the environment and returns the first observation, or the first\
         (state, obs) tuple.

        Args:
            return_state: If true return a also the initial state of the env.

        Returns:
            Observation of the environment if `return_state` is False. Otherwise,
            return (state, obs) after reset.
        """
        time_step = self.gym_env.reset()
        observed = self._time_step_to_obs(time_step)
        self._render_i = 0
        if not return_state:
            return observed
        else:
            return self.get_state(), observed

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

    def step_with_dt(self, action: Union[np.ndarray, int, float], dt: int = 1) -> tuple:
        """
         Take ``dt`` simulation steps and make the environment evolve in multiples\
          of ``self.frameskip`` for a total of ``dt`` * ``self.frameskip`` steps.

        Args:
            action: Chosen action applied to the environment.
            dt: Consecutive number of times that the action will be applied.

        Returns:
            if state is None returns ``(observs, reward, terminal, info)``
            else returns ``(new_state, observs, reward, terminal, info)``

        """
        reward = 0
        obs, lost_live, terminal, oob = None, False, False, False
        info = {"lives": -1}
        n_steps = 0

        for _ in range(int(dt)):
            end = False
            for _ in range(self.frameskip):
                time_step = self.gym_env.step(action)
                end = end or time_step.last()
                reward += time_step.reward if time_step.reward is not None else 0.0
                custom_terminal = self.custom_terminal_condition(info, {}, end)
                terminal = terminal or oob or custom_terminal
                terminal = (terminal or lost_live) if self.episodic_life else terminal
                n_steps += 1
                if terminal:
                    break
            if terminal:
                break
        obs = self._time_step_to_obs(time_step)
        # This allows to get the original values even when using an episodic life environment
        info["terminal"] = terminal
        info["lost_live"] = lost_live
        info["oob"] = oob
        info["win"] = self.get_win_condition(info)
        info["n_steps"] = n_steps
        return obs, reward, terminal, info

    @staticmethod
    def _time_step_to_obs(time_step) -> np.ndarray:
        # Concat observations in a single array, so it is easier to calculate distances
        obs_array = np.hstack(
            [np.array([time_step.observation[x]]).flatten() for x in time_step.observation],
        )
        return obs_array


class __CustomDeath:
    """DEPRECATED: Class for taking into account arbitrary boundary conditions."""

    def __init__(self, domain_name="cartpole", task_name="balance"):
        self._domain_name = domain_name
        self._task_name = task_name

    @property
    def task_name(self):
        return self._task_name

    @property
    def domain_name(self):
        return self._domain_name

    def calculate(self, env: DMControlEnv, time_step, last_time_step):

        if self._domain_name == "cartpole" and self.task_name == "balance":
            return self._cartpole_balance_death(env=env, time_step=time_step)
        elif self._domain_name == "hopper":
            return self._hopper_death(env=env, time_step=time_step, last_time_step=last_time_step)
        elif self._domain_name == "walker":
            return self._walker_death(env=env, time_step=time_step, last_time_step=last_time_step)
        else:
            return self._default_death(time_step, last_time_step)

    @staticmethod
    def _default_death(time_step, last_time_step) -> bool:
        last_rew = last_time_step.reward if last_time_step is not None else 0
        return time_step.reward <= 0 and last_rew > 0

    @staticmethod
    def _cartpole_balance_death(env, time_step) -> bool:
        """
        If the reward is less than 0.7 consider a state dead.

        This threshold is because rewards lesser than 0.7 involve positions where
         the cartpole is not balanced.
        """
        return time_step.reward < 0.75 or abs(env.physics.cart_position()) > 0.5

    @staticmethod
    def _hopper_death(env, time_step, last_time_step) -> bool:
        # min_torso_height = 0.1
        # max_reward_drop = 0.3

        # torso_touches_ground = env.physics.height() < min_torso_height
        # reward_change = time_step.reward - (last_time_step.reward if
        # last_time_step is not None else 0)
        # reward_drops = reward_change < -max_reward_drop * env.dt
        return False

    @staticmethod
    def _walker_death(env, time_step, last_time_step) -> bool:
        min_torso_height = 0.1
        max_reward_drop_pct = 0.5
        # max_tilt = 0
        min_reward = 0.1

        torso_touches_ground = env.physics.torso_height() < min_torso_height
        last_reward = last_time_step.reward if last_time_step is not None else 0.00001
        reward_change = time_step.reward / last_reward
        reward_drops = reward_change < max_reward_drop_pct
        torso_very_tilted = False  # abs(env.physics.torso_upright())
        # < max_tilt and reward_change < 0
        # torso_very_tilted = torso_very_tilted if not env.state.dead else False

        crappy_reward = time_step.reward < min_reward  # if not env.state.dead else False

        return reward_drops or torso_touches_ground or torso_very_tilted or crappy_reward
