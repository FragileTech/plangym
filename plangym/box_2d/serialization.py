import copy

import numpy

from Box2D.Box2D import b2Vec2, b2Transform


def get_body_attributes(body):
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


def serialize(value):
    if isinstance(value, b2Vec2):
        return tuple([*value.copy()])
    elif isinstance(value, b2Transform):
        return {
            "angle": float(value.angle),
            "position": tuple([*value.position.copy()]),
        }
    else:
        return copy.copy(value)


def serialize_body_state(state_dict):
    return {k: serialize(v) for k, v in state_dict.items()}


def set_value_to_body(body, name, value):
    body_object = getattr(body, name)
    if isinstance(body_object, b2Vec2):
        return body_object.Set(*value)
    elif isinstance(body_object, b2Transform):
        body_object.angle = value["angle"]
        body_object.position.Set(*value["position"])
    else:
        return setattr(body, name, value)


def set_body_state(body, state):
    state = state[0] if isinstance(state, numpy.ndarray) else state
    for k, v in state.items():
        set_value_to_body(body, k, v)
    return body


def get_body_state(body):
    data = get_body_attributes(body)
    return serialize_body_state(data)


def get_world_state(world):
    return [get_body_state(b) for b in world.bodies]


def set_world_state(world, state):
    for body, state in zip(world.bodies, state):
        set_body_state(body, state)


def get_env_state(env):
    return get_world_state(env.unwrapped.world)


def set_env_state(env, state):
    return set_world_state(env.unwrapped.world, state)
