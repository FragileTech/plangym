import copy
import math
from typing import Any, Dict, Iterable

import numpy as numpy
from gym.envs.box2d.lunar_lander import (
    FPS,
    LunarLander as GymLunarLander,
    SCALE,
    MAIN_ENGINE_POWER,
    LEG_DOWN,
    SIDE_ENGINE_POWER,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    VIEWPORT_H,
    VIEWPORT_W,
)

from plangym.classic_control import ClassicControl
from plangym.core import GymEnvironment, wrap_callable
from plangym.box_2d.serialization import get_env_state, set_env_state
import math
import numpy as np
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
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

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class FastGymLunarLander(GymLunarLander):

    def __init__(self, deterministic: bool=False):
        self.deterministic = deterministic
        super(FastGymLunarLander, self).__init__()


    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        deterministic_terrain = True # self.deterministic
        height = np.ones(CHUNKS+1) * H /4 if deterministic_terrain else self.np_random.uniform(0, H/2, size=(CHUNKS+1,) )
        chunk_x  = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y  = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i],   smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append( [p1, p2, (p2[0],H), (p1[0],H)] )

        self.moon.color1 = (0.0,0.0,0.0)
        self.moon.color2 = (0.0,0.0,0.0)

        initial_y = VIEWPORT_H/SCALE
        self.lander = self.world.CreateDynamicBody(
            position = (VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander.color1 = (0.5,0.4,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        init_force_x = 0. if self.deterministic else self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        init_force_y = 0. if self.deterministic else self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        self.lander.ApplyForceToCenter( (init_force_x, init_force_y), True)

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0,0]) if self.continuous else 0)[0]

    def step(self, action):
        if self.continuous:
            action = numpy.clip(action, -1, +1).astype(numpy.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [0, 0] if self.deterministic else [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
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
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and numpy.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
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
        if not self.lander.awake:
            done = True
            reward = +100
        self.prev_reward = reward
        return numpy.array(state, dtype=numpy.float32), reward, done, {}


class LunarLander(GymEnvironment):
    def __init__(
        self,
        name: str = None,
        dt: int = 1,
        min_dt: int = 1,
        episodic_live: bool = True,
        autoreset: bool = True,
        wrappers: Iterable[wrap_callable] = None,
        delay_init: bool = False,
        states_on_reset: bool = True,
        deterministic: bool = False,
    ):
        self.deterministic = deterministic
        super(LunarLander, self).__init__(
            name="Lunarlander-plangym",
            dt=dt,
            min_dt=min_dt,
            episodic_live=episodic_live,
            autoreset=autoreset,
            wrappers=wrappers,
            delay_init=delay_init,
            states_on_reset=states_on_reset,
        )

    def init_env(self):
        """Initialize the target :class:`gym.Env` instance."""
        self.gym_env = FastGymLunarLander(deterministic=self.deterministic)
        if self._wrappers is not None:
            self.apply_wrappers(self._wrappers)
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space
        self.reward_range = self.gym_env.reward_range
        self.metadata = self.gym_env.metadata

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
        state = get_env_state(self.gym_env) + env_data
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
        set_env_state(self.gym_env, box_2d_state)
        self.gym_env.lander.awake = state[0][-6]
        self.gym_env.game_over = state[0][-5]
        self.gym_env.prev_shaping = state[0][-4]
        self.gym_env.prev_reward = state[0][-3]
        self.gym_env.legs[0].ground_contact = state[0][-2]
        self.gym_env.legs[1].ground_contact = state[0][-1]

    @staticmethod
    def get_win_condition(info: Dict[str, Any]) -> bool:
        """Return ``True`` if the current state corresponds to winning the game."""
        return False

    def _lunar_lander_end(self, obs):
        if self.gym_env.game_over or abs(obs[0]) >= 1.0 or obs[1]< 0.:
            return True
        elif not self.gym_env.lander.awake:
            return True
        return False

    def _step_with_dt(self, action, dt):
        obs, reward, end, info = super(LunarLander, self)._step_with_dt(action, dt)
        terminal = self._lunar_lander_end(obs)
        fin = terminal or end
        #reward /= (obs[0] ** 2 + 0.25 * obs[1] ** 2+ 1e-4)
        if self.gym_env.game_over or (abs(obs[0]) >= 1.0 and (abs(obs[1]) < 0.5 * 1e-3)):
            fin = True
            reward += -100
        if not self.gym_env.lander.awake or (abs(obs[0]) >= 1.0 and (abs(obs[1]) < 0.5 * 1e-3 and obs[1]>=0.)):
            fin = True
            reward += 100
            info["win"] = True
        info["oob"] = fin

        return obs, reward, end, info


class BipedalWalker(GymEnvironment):

    def get_state(self) -> numpy.ndarray:
        """
        Recover the internal state of the simulation.

        An state must completely describe the Environment at a given moment.
        """
        state = get_env_state(self.gym_env)
        return numpy.array((state, None), dtype=object)

    def set_state(self, state: numpy.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        box_2d_state = state[0]
        set_env_state(self.gym_env, box_2d_state)
