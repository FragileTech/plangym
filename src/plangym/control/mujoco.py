"""Implement the plangym API for gymnasium[mujoco] environments."""

from typing import Iterable

import gymnasium as gym
import mujoco
import numpy

from plangym.core import PlangymEnv, wrap_callable


class MujocoEnv(PlangymEnv):
    """Wrap gymnasium[mujoco] environments for planning problems.

    Gymnasium's MuJoCo environments use the MuJoCo physics engine for
    simulating articulated body dynamics.

    For more information about the environments, please refer to:
    https://gymnasium.farama.org/environments/mujoco/
    """

    DEFAULT_OBS_TYPE = "coords"

    def __init__(
        self,
        name: str = "Ant-v4",
        frameskip: int = 1,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] | None = None,
        delay_setup: bool = False,
        render_mode: str = "rgb_array",
        obs_type: str | None = None,
        return_image: bool = False,
        remove_time_limit: bool = False,  # noqa: ARG002
        **kwargs,
    ):
        """Initialize a MujocoEnv.

        Args:
            name: Name of the MuJoCo environment (e.g., "Ant-v4", "HalfCheetah-v4").
            frameskip: Number of times to repeat action.
            autoreset: Restart environment when reaching a terminal state.
            wrappers: Wrappers to apply to the underlying gym env.
            delay_setup: If True, delay environment initialization.
            render_mode: Rendering mode (None, "human", "rgb_array").
            obs_type: Observation type ("coords", "rgb", "grayscale").
            return_image: If True, add "rgb" key to observation dict.
            remove_time_limit: Ignored (MuJoCo envs handle time limits internally).
            **kwargs: Additional arguments passed to gym.make().

        """
        self._init_kwargs = kwargs
        super().__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            wrappers=wrappers,
            delay_setup=delay_setup,
            render_mode=render_mode,
            obs_type=obs_type,
            return_image=return_image,
        )

    def init_gym_env(self) -> gym.Env:
        """Initialize the gymnasium MuJoCo environment."""
        return gym.make(
            self.name,
            render_mode=self.render_mode,
            disable_env_checker=True,
            **self._init_kwargs,
        )

    def setup(self):
        """Initialize the gym environment and reset it before initializing spaces."""
        super().setup()

    def init_spaces(self):
        """Initialize spaces after resetting to avoid render order issues."""
        # Reset the environment first to allow rendering
        self.gym_env.reset()
        super().init_spaces()

    def get_state(self) -> numpy.ndarray:
        """Get the full integration state including time, qpos, qvel, and solver state.

        Uses mjSTATE_INTEGRATION which captures all state needed for deterministic
        simulation continuation, including constraint solver warmstart values.
        """
        env = self.gym_env.unwrapped
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state_size = mujoco.mj_stateSize(env.model, spec)
        state = numpy.zeros(state_size)
        mujoco.mj_getState(env.model, env.data, state, spec)
        return state

    def set_state(self, state: numpy.ndarray) -> None:
        """Set the full integration state and update derived quantities.

        Uses mjSTATE_INTEGRATION and calls mj_forward to recompute all derived
        quantities (Cartesian positions, velocities, etc.) for deterministic
        simulation after state restoration.
        """
        env = self.gym_env.unwrapped
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(env.model, env.data, state, spec)
        mujoco.mj_forward(env.model, env.data)

    def get_image(self) -> numpy.ndarray:
        """Return rendered RGB image of the environment."""
        return self.gym_env.render()
