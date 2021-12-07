"""Implement the ``plangym`` API for Box2D environments."""
import copy

from Box2D.Box2D import b2Transform, b2Vec2
import numpy

from plangym.core import PlanEnvironment


class Box2DState:
    """Extract state information from Box2D environments."""

    @staticmethod
    def get_body_attributes(body) -> dict:
        """Return a dictionary containing the of all attributes of a given body."""
        base = {
            "mass": body.mass,
            "inertia": body.inertia,
            "localCenter": body.localCenter,
        }
        state_info = {
            "type": body.type,
            "bullet": body.bullet,
            "awake": body.awake,
            "sleepingAllowed": body.sleepingAllowed,
            "active": body.active,
            "fixedRotation": body.fixedRotation,
        }
        kinematics = {"transform": body.transform, "position": body.position, "angle": body.angle}
        other = {
            "worldCenter": body.worldCenter,
            "localCenter": body.localCenter,
            "linearVelocity": body.linearVelocity,
            "angularVelocity": body.angularVelocity,
        }
        base.update(kinematics)
        base.update(state_info)
        base.update(other)
        return base

    @staticmethod
    def serialize_body_attribute(value):
        """Copy one body attribute."""
        if isinstance(value, b2Vec2):
            return tuple([*value.copy()])
        elif isinstance(value, b2Transform):
            return {
                "angle": float(value.angle),
                "position": tuple([*value.position.copy()]),
            }
        else:
            return copy.copy(value)

    @classmethod
    def serialize_body_state(cls, state_dict):
        """Serialize the state of the target body data."""
        return {k: cls.serialize_body_attribute(v) for k, v in state_dict.items()}

    @staticmethod
    def set_value_to_body(body, name, value):
        """Set the target value to a body attribute."""
        body_object = getattr(body, name)
        if isinstance(body_object, b2Vec2):
            return body_object.Set(*value)
        elif isinstance(body_object, b2Transform):
            body_object.angle = value["angle"]
            body_object.position.Set(*value["position"])
        else:
            return setattr(body, name, value)

    @classmethod
    def set_body_state(cls, body, state):
        """Set the state to the target body."""
        state = state[0] if isinstance(state, numpy.ndarray) else state
        for k, v in state.items():
            cls.set_value_to_body(body, k, v)
        return body

    @classmethod
    def serialize_body(cls, body):
        """Serialize the data of the target body instance."""
        data = cls.get_body_attributes(body)
        return cls.serialize_body_state(data)

    @classmethod
    def serialize_world_state(cls, world):
        """Serialize the state of all the bodies in world."""
        return [cls.serialize_body(b) for b in world.bodies]

    @classmethod
    def set_world_state(cls, world, state):
        """Set the state of world to the provided state."""
        for body, state in zip(world.bodies, state):
            cls.set_body_state(body, state)

    @classmethod
    def get_env_state(cls, env):
        """Get the serialized state of the target environment."""
        return cls.serialize_world_state(env.unwrapped.world)

    @classmethod
    def set_env_state(cls, env, state):
        """Set the serialized state to the target environment."""
        return cls.set_world_state(env.unwrapped.world, state)


class Box2DEnv(PlanEnvironment):
    """Common interface for working with Box2D environments."""

    def get_state(self) -> numpy.array:
        """
        Recover the internal state of the simulation.

        An state must completely describe the Environment at a given moment.
        """
        state = Box2DState.get_env_state(self.gym_env)  # pickle.dumps(get_env_state(self.gym_env))

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
        Box2DState.set_env_state(self.gym_env, state[0])

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
