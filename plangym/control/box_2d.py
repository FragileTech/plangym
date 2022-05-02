"""Implement the ``plangym`` API for Box2D environments."""
import copy

import numpy

from plangym.core import PlangymEnv


class Box2DState:
    """
    Extract state information from Box2D environments.

    This class implements basic functionalities to get the necessary
    elements to construct a Box2D state.
    """

    @staticmethod
    def get_body_attributes(body) -> dict:
        """
        Return a dictionary containing the attributes of a given body.

        Given a ``Env.world.body`` element, this method constructs a dictionary
        whose entries describe all body attributes.
        """
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
        from Box2D.Box2D import b2Transform, b2Vec2

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
        """
        Serialize the state of the target body data.

        This method takes as argument the result given by the method
        ``self.get_body_attributes``, the latter consisting in a dictionary
        containing all attribute elements of a body. The method returns
        a dictionary whose values are the serialized attributes of the body.
        """
        return {k: cls.serialize_body_attribute(v) for k, v in state_dict.items()}

    @staticmethod
    def set_value_to_body(body, name, value):
        """Set the target value to a body attribute."""
        from Box2D.Box2D import b2Transform, b2Vec2

        body_object = getattr(body, name)
        if isinstance(body_object, b2Vec2):
            body_object.Set(*value)
        elif isinstance(body_object, b2Transform):
            body_object.angle = value["angle"]
            body_object.position.Set(*value["position"])
        else:
            setattr(body, name, value)

    @classmethod
    def set_body_state(cls, body, state):
        """
        Set the state to the target body.

        The method defines the corresponding body attribute to the value
        selected by the user.
        """
        state = state[0] if isinstance(state, numpy.ndarray) else state
        for k, v in state.items():
            cls.set_value_to_body(body, k, v)
        return body

    @classmethod
    def serialize_body(cls, body):
        """Serialize the data of the target ``Env.world.body`` instance."""
        data: dict = cls.get_body_attributes(body)
        return cls.serialize_body_state(data)

    @classmethod
    def serialize_world_state(cls, world):
        """
        Serialize the state of all the bodies in world.

        The method serializes all body elements contained within the
        given ``Env.world`` object.
        """
        return [cls.serialize_body(b) for b in world.bodies]

    @classmethod
    def set_world_state(cls, world, state):
        """
        Set the state of the world environment to the provided state.

        The method states the current environment by defining its world
        bodies' attributes.
        """
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


class Box2DEnv(PlangymEnv):
    """Common interface for working with Box2D environments released by `gym`."""

    def get_state(self) -> numpy.array:
        """
        Recover the internal state of the simulation.

        A state must completely describe the Environment at a given moment.
        """
        state = Box2DState.get_env_state(self.gym_env)
        return numpy.array((state, None), dtype=object)

    def set_state(self, state: numpy.ndarray) -> None:
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None

        """
        Box2DState.set_env_state(self.gym_env, state[0])
