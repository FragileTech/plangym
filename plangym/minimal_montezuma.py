from collections import Counter, defaultdict
import typing
import logging

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np

from plangym.env import AtariEnvironment

# ------------------------------------------------------------------------------

# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
class IgnoreNoHandles(logging.Filter):
    def filter(self, record):
        if record.getMessage() == "No handles with labels found to put in legend.":
            return 0
        return 1


from PIL import Image

_plt_logger = logging.getLogger("matplotlib.legend")
_plt_logger.addFilter(IgnoreNoHandles())


class MontezumaPosLevel:
    __slots__ = ["level", "score", "room", "x", "y", "tuple"]

    def __init__(self, level, score, room, x, y):
        self.level = level
        self.score = score
        self.room = room
        self.x = x
        self.y = y

        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.level, self.score, self.room, self.x, self.y)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, MontezumaPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        return f"Level={self.level} Room={self.room} Objects={self.score} x={self.x} y={self.y}"


def resize_frame(frame: np.ndarray, height: int, width: int, mode="RGB") -> np.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        height: Height of the resized image.
        width: Width of the resized image.

    Returns:
        The resized frame that matches the provided width and height.
    """
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize((height, width))
    return np.array(frame)[:, :, None]


def convert_state(state):
    return (
        (
            resize_frame(
                state, height=MyMontezuma.TARGET_SHAPE[0], width=MyMontezuma.TARGET_SHAPE[1]
            )
            / 255.0
        )
        * MyMontezuma.MAX_PIX_VALUE
    ).astype(np.uint8)


PYRAMID = [
    [-1, -1, -1, 0, 1, 2, -1, -1, -1],
    [-1, -1, 3, 4, 5, 6, 7, -1, -1],
    [-1, 8, 9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23],
]

OBJECT_PIXELS = [
    50,  # Hammer/mallet
    40,  # Key 1
    40,  # Key 2
    40,  # Key 3
    37,  # Sword 1
    37,  # Sword 2
    42,  # Torch
]

KNOWN_XY = [None] * 24

KEY_BITS = 0x8 | 0x4 | 0x2


def get_room_xy(room):
    if KNOWN_XY[room] is None:
        for y, l in enumerate(PYRAMID):
            if room in l:
                KNOWN_XY[room] = (l.index(room), y)
                break
    return KNOWN_XY[room]


class MyMontezuma:
    def __init__(
        self,
        check_death: bool = True,
        unprocessed_state: bool = False,
        score_objects: bool = False,
        x_repeat=2,
        objects_from_pixels=False,
        objects_remember_rooms=False,
        only_keys=False,
    ):  # TODO: version that also considers the room objects were found in
        self.env = gym.make("MontezumaRevengeDeterministic-v4")
        self.env.reset()
        self.rooms = {}
        self.unwrapped.seed(0)
        self.unprocessed_state = unprocessed_state
        self.pos = MontezumaPosLevel(0, 0, 0, 0, 0)
        self.x_repeat = x_repeat

    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        unprocessed_state = self.env.reset()
        for _ in range(3):
            unprocessed_state = self.env.step(0)[0]
        self.pos = self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed_state))
        if self.get_pos().room not in self.rooms:
            self.rooms[self.get_pos().room] = (
                False,
                unprocessed_state[50:].repeat(self.x_repeat, axis=1),
            )
        return self.get_observation(unprocessed_state)

    def pos_from_unprocessed_state(self, face_pixels):
        face_pixels = [(y, x * self.x_repeat) for y, x in face_pixels]
        if len(face_pixels) == 0:
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        room = 1
        level = 0
        if self.pos is not None:
            room = self.pos.room
            level = self.pos.level
            direction_x = np.clip(int((self.pos.x - x) / 50), -1, 1)
            direction_y = np.clip(int((self.pos.y - y) / 50), -1, 1)
            if direction_x != 0 or direction_y != 0:
                # TODO(AE): shoudln't this call the static method?
                room_x, room_y = get_room_xy(self.pos.room)
                if room == 15 and room_y + direction_y >= len(PYRAMID):
                    room = 1
                    level += 1
                elif direction_x == 0 or direction_y == 0:
                    _room = PYRAMID[room_y + direction_y][room_x + direction_x]
                    # if _room != -1:
                    room = room
                    assert room != -1, f"Impossible room change: ({direction_y}, {direction_x})"

        score = 0
        return MontezumaPosLevel(level, score, room, x, y)

    def get_observation(self, observation) -> np.ndarray:
        # obs = resize_frame(observation[25:185, :], width=45, height=45, mode="L")
        pos = np.array([self.pos.x, self.pos.y, self.pos.room])  # np.array(self.pos.tuple +
        # self.room_time)
        return np.concatenate([observation.flatten(), pos])

    def get_restore(self):
        return self.unwrapped.clone_full_state(), self.pos

    def restore(self, data):
        full_state, pos = data
        self.env.reset()
        self.unwrapped.restore_full_state(full_state.copy())
        self.pos = pos
        return

    @staticmethod
    def get_face_pixels(unprocessed_state):
        # TODO: double check that this color does not re-occur somewhere else
        # in the environment.
        return set(zip(*np.where(unprocessed_state[50:, :, 0] == 228)))

    def step(self, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        unprocessed_state, reward, done, lol = self.env.step(action)

        face_pixels = self.get_face_pixels(unprocessed_state)

        self.pos = self.pos_from_unprocessed_state(face_pixels)
        if self.pos.room not in self.rooms:
            self.rooms[self.pos.room] = (
                False,
                unprocessed_state[50:].repeat(self.x_repeat, axis=1),
            )
        obs = self.get_observation(unprocessed_state)
        # print(obs, done)
        return obs, reward, done, lol

    def get_pos(self):
        assert self.pos is not None
        return self.pos

    @staticmethod
    def get_room_xy(room):
        if KNOWN_XY[room] is None:
            for y, l in enumerate(PYRAMID):
                if room in l:
                    KNOWN_XY[room] = (l.index(room), y)
                    break
        return KNOWN_XY[room]

    @staticmethod
    def get_room_out_of_bounds(room_x, room_y):
        return room_y < 0 or room_x < 0 or room_y >= len(PYRAMID) or room_x >= len(PYRAMID[0])

    @staticmethod
    def get_room_from_xy(room_x, room_y):
        return PYRAMID[room_y][room_x]

    @staticmethod
    def make_pos(score, pos):
        return MontezumaPosLevel(pos.level, score, pos.room, pos.x, pos.y)


# ------------------------------------------------------------------------------


class Montezuma(AtariEnvironment):
    def __init__(
        self,
        n_repeat_action: int = 1,
        min_dt: int = 1,
        episodic_live: bool = False,
        autoreset: bool = True,
        name=None,
        *args,
        **kwargs
    ):

        super(Montezuma, self).__init__(
            name="MontezumaRevengeDeterministic-v4",
            n_repeat_action=n_repeat_action,
            clone_seeds=True,
            min_dt=min_dt,
            obs_ram=False,
            episodic_live=episodic_live,
            autoreset=autoreset,
        )
        self._env = MyMontezuma(*args, **kwargs)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.reward_range = self._env.reward_range
        self.metadata = self._env.metadata

    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def n_actions(self):
        return self._env.action_space.n

    def get_state(self) -> np.ndarray:
        """
        Recover the internal state of the simulation. If clone seed is False the
        environment will be stochastic.
        Cloning the full state ensures the environment is deterministic.
        """
        data = self._env.get_restore()
        (full_state, pos) = data
        array = np.concatenate([full_state, np.array(pos.tuple)]).copy()
        return array

    def set_state(self, state: np.ndarray):
        """
        Set the internal state of the simulation.

        Args:
            state: Target state to be set in the environment.

        Returns:
            None
        """
        pos_vals = state[-5:].tolist()
        pos = MontezumaPosLevel(
            level=int(pos_vals[0]),
            score=float(pos_vals[1]),
            room=int(pos_vals[2]),
            x=float(pos_vals[3]),
            y=float(pos_vals[4]),
        )
        full_state = state[:-5].copy().astype(np.uint8)
        data = (full_state, pos)
        self._env.restore(data)

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None
    ) -> tuple:
        """

        Take n_repeat_action simulation steps and make the environment evolve
        in multiples of min_dt.
        The info dictionary will contain a boolean called 'lost_live' that will
        be true if a life was lost during the current step.

        Args:
            action: Chosen action applied to the environment.
            state: Set the environment to the given state before stepping it.
            n_repeat_action: Consecutive number of times that the action will be applied.

        Returns:
            if states is None returns (observs, rewards, ends, infos)
            else returns(new_states, observs, rewards, ends, infos)
        """
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        if state is not None:
            self.set_state(state)
        reward = 0
        _end, lost_live = False, False
        info = {"lives": -1}
        terminal = False
        game_end = False
        for _ in range(int(n_repeat_action)):
            for _ in range(self.min_dt):
                obs, _reward, _end, _info = self._env.step(action)
                _info["lives"] = _info.get("ale.lives", -1)
                lost_live = info["lives"] > _info["lives"] or lost_live
                game_end = game_end or _end
                terminal = terminal or game_end
                terminal = terminal or lost_live if self.episodic_life else terminal
                info = _info.copy()
                reward += _reward
                if _end:
                    break
            if _end:
                break
        # This allows to get the original values even when using an episodic life environment
        info["terminal"] = terminal
        info["lost_live"] = lost_live
        info["game_end"] = game_end
        if state is not None:
            new_state = self.get_state()
            data = new_state, obs, reward, terminal, info
        else:
            data = obs, reward, terminal, info
        if _end and self.autoreset:
            self._env.reset()
        return data

    def render(self):
        """Render the environment using OpenGL. This wraps the OpenAI render method."""
        return self._env.render()
