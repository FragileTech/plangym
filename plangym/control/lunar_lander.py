"""Implementation of LunarLander with no fire coming out of the engines that steps faster."""
import copy
import math
from typing import Iterable, Optional

import numpy

from plangym.control.box_2d import Box2DState
from plangym.core import PlangymEnv, wrap_callable


try:
    from Box2D.b2 import edgeShape, fixtureDef, polygonShape, revoluteJointDef
    from gym.envs.box2d.lunar_lander import ContactDetector, LunarLander as GymLunarLander

    import_error = None
except ImportError as e:
    import_error = e
    GymLunarLander = object

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discrete actions
# (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two
# numbers in the state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is
# about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the
# lander crashes or comes to rest, receiving additional -100 or +100 points.
# Each leg ground contact is +10.
# Firing main engine is -0.3 points each frame.
# Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite,
# so an agent can learn to fly and then land on its first attempt.
# Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class FastGymLunarLander(GymLunarLander):
    """Faster implementation of the LunarLander without bells and whistles."""

    FPS = FPS

    def __init__(self, deterministic: bool = False, continuous: bool = False):
        """Initialize a :class:`FastGymLunarLander``."""
        self.deterministic = deterministic
        self.game_over = False
        self.prev_shaping = None
        self.helipad_x1 = None
        self.helipad_x2 = None
        self.helipad_y = None
        self.moon = None
        self.sky_polys = None
        self.lander = None
        self.legs = None
        self.drawlist = None
        self.viewer = None
        self.moon = None
        self.lander = None
        self.particles = None
        self.prev_reward = None
        self.observation_space = None
        self.action_space = None
        self.continuous = continuous
        super(FastGymLunarLander, self).__init__()

    def reset(self) -> tuple:
        """Reset the environment to its initial state."""
        # Reset environment data
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        # Define environment bodies
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        # terrain shape
        CHUNKS = 11
        height = (
            numpy.ones(CHUNKS + 1) * H / 4
            if self.deterministic
            else self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        )
        # Define helipad
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]
        # Define moon
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)
        # Define lander body and initial position
        initial_y = VIEWPORT_H / SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        # Unlike in the original LunarLander, the initial force can be deterministic.
        init_force_x = (
            0.0 if self.deterministic else self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        )
        init_force_y = (
            0.0 if self.deterministic else self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        )
        self.lander.ApplyForceToCenter((init_force_x, init_force_y), True)
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        self.drawlist = [self.lander] + self.legs

        return self.step(numpy.array([0, 0]) if self.continuous else 0)[0]

    def step(self, action: int) -> tuple:
        """Step the environment applying the provided action."""
        if self.continuous:
            action = numpy.clip(action, -1, +1).astype(numpy.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = (
            [0, 0]
            if self.deterministic
            else [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        )
        # Main engine
        m_power = 0.0
        fire_me_continuous = self.continuous and action[0] > 0.0
        fire_me_discrete = not self.continuous and action == 2
        fire_main_engine = fire_me_continuous or fire_me_discrete
        if fire_main_engine:

            if self.continuous:
                m_power = (numpy.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = (
                tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            )  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            # We do not create any decorative particle.
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
        # Orientation engines
        s_power = 0.0
        fire_oe_continuous = self.continuous and numpy.abs(action[1]) > 0.5
        fire_oe_discrete = not self.continuous and action in [1, 3]
        fire_orientation_engine = fire_oe_continuous or fire_oe_discrete
        if fire_orientation_engine:
            if self.continuous:
                direction = numpy.sign(action[1])
                s_power = numpy.clip(numpy.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * numpy.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * numpy.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:  # pragma: no cover
            done = True
            reward = +100
        self.prev_reward = reward
        self.game_over = done or self.game_over
        return numpy.array(state, dtype=numpy.float32), reward, done, {}

    def render(self, mode="human"):
        """Render the environment."""
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon(
                [
                    (x, flagy2),
                    (x, flagy2 - 10 / SCALE),
                    (x + 25 / SCALE, flagy2 - 5 / SCALE),
                ],
                color=(0.8, 0.8, 0),
            )

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class LunarLander(PlangymEnv):
    """Fast LunarLander that follows the plangym API."""

    def __init__(
        self,
        name: str = None,
        frameskip: int = 1,
        episodic_life: bool = True,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_setup: bool = False,
        deterministic: bool = False,
        continuous: bool = False,
        render_mode: Optional[str] = None,
        remove_time_limit=None,
        **kwargs,
    ):
        """Initialize a :class:`LunarLander`."""
        self._deterministic = deterministic
        self._continuous = continuous
        super(LunarLander, self).__init__(
            name="LunarLander-plangym",
            frameskip=frameskip,
            episodic_life=episodic_life,
            autoreset=autoreset,
            wrappers=wrappers,
            delay_setup=delay_setup,
            render_mode=render_mode,
            **kwargs,
        )

    @property
    def deterministic(self) -> bool:
        """Return true if the LunarLander simulation is deterministic."""
        return self._deterministic

    @property
    def continuous(self) -> bool:
        """Return true if the LunarLander agent takes continuous actions as input."""
        return self._continuous

    def init_gym_env(self) -> FastGymLunarLander:
        """Initialize the target :class:`gym.Env` instance."""
        if import_error is not None:
            raise import_error
        gym_env = FastGymLunarLander(
            deterministic=self.deterministic,
            continuous=self.continuous,
        )
        gym_env.reset()
        return gym_env

    def get_state(self) -> numpy.ndarray:
        """
        Recover the internal state of the simulation.

        An state must completely describe the Environment at a given moment.
        """
        env_data = [
            bool(self.gym_env.lander.awake),
            bool(self.gym_env.game_over),
            copy.copy(self.gym_env.prev_shaping),
            copy.copy(self.gym_env.prev_reward),
            bool(self.gym_env.legs[0].ground_contact),
            bool(self.gym_env.legs[1].ground_contact),
        ]
        state = Box2DState.get_env_state(self.gym_env) + env_data
        return numpy.array((state, None), dtype=object)

    def set_state(self, state: numpy.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        box_2d_state = state[0][:-6]
        Box2DState.set_env_state(self.gym_env, box_2d_state)
        self.gym_env.lander.awake = state[0][-6]
        self.gym_env.game_over = state[0][-5]
        self.gym_env.prev_shaping = state[0][-4]
        self.gym_env.prev_reward = state[0][-3]
        self.gym_env.legs[0].ground_contact = state[0][-2]
        self.gym_env.legs[1].ground_contact = state[0][-1]

    def process_terminal(self, terminal, obs=None, **kwargs) -> bool:
        """Return the terminal condition considering the lunar lander state."""
        obs = [0] if obs is None else obs
        end = (
            self.gym_env.game_over
            or (self.obs_type == "coords" and abs(obs[0]) >= 1.0)
            or not self.gym_env.lander.awake
        )
        return terminal or end
