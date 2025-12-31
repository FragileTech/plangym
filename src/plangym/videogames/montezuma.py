"""Implementation of the montezuma environment adapted for planning problems."""

from typing import Iterable, Any, SupportsFloat

import cv2
import gymnasium as gym
import numpy
from numpy import ndarray

from plangym.core import wrap_callable
from plangym.utils import remove_time_limit
from plangym.videogames.atari import AtariEnv, ale_to_ram


class MontezumaPosLevel:
    """Contains the information of Panama Joe."""

    __slots__ = ["level", "room", "score", "tuple", "x", "y"]

    def __init__(self, level, score, room, x, y):
        """Initialize a :class:`MontezumaPosLevel`."""
        self.level = level
        self.score = score
        self.room = room
        self.x = x
        self.y = y
        self.tuple = None
        self.set_tuple()

    def set_tuple(self):
        """Set the tuple values from the other attributes."""
        self.tuple = (self.level, self.score, self.room, self.x, self.y)

    def __hash__(self):
        """Hash the current instance."""
        return hash(self.tuple)

    def __eq__(self, other):
        """Compare equality between instances."""
        if not isinstance(other, MontezumaPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        """Return the tuple containing the attributes."""
        return self.tuple

    def __setstate__(self, d):
        """Set the class attributes using the provided tuple."""
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        """Print the attributes of the current instance."""
        return f"Level={self.level} Room={self.room} Objects={self.score} x={self.x} y={self.y}"


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

KNOWN_XY: list[None | tuple[int, int]] = [None] * 24

KEY_BITS = 0x8 | 0x4 | 0x2


class CustomMontezuma:
    """MontezumaEnv environment that tracks the room and position of Panama Joe."""

    def __init__(
        self,
        check_death: bool = True,
        obs_type: str = "rgb",
        score_objects: bool = False,
        objects_from_pixels: bool = False,
        objects_remember_rooms: bool = False,
        only_keys: bool = False,
        death_room_8: bool = True,
        render_mode: str = "rgb_array",
        x_repeat: int = 1,
    ):  # TODO: version that also considers the room objects were found in
        """Initialize a :class:`CustomMontezuma`."""
        self.render_mode = render_mode
        self.score_objects = score_objects
        self.check_death = check_death
        self.objects_from_pixels = objects_from_pixels
        self.objects_remember_rooms = objects_remember_rooms
        self.only_keys = only_keys
        self.coords_obs = obs_type == "coords"
        self._x_repeat = x_repeat
        self._death_room_8 = death_room_8

        env = gym.make("MontezumaRevengeDeterministic-v4", render_mode=self.render_mode)
        self.env = remove_time_limit(env)
        self.unwrapped.seed(0)
        self.env.reset()
        self.ram = None
        self.cur_steps = 0
        self.cur_score = 0
        self.rooms = {}
        self.cur_lives = 5
        self.pos = MontezumaPosLevel(0, 0, 0, 0, 0)
        if self.coords_obs:
            shape = self.get_coords().shape
            self.observation_space = gym.spaces.Box(
                low=-numpy.inf,
                high=numpy.inf,
                dtype=numpy.float32,
                shape=shape,
            )

    @staticmethod
    def get_room_xy(room: int) -> None | tuple[int, int]:
        """Get the tuple that encodes the provided room."""
        if room >= len(KNOWN_XY) or room < 0:
            return None
        if KNOWN_XY[room] is None:
            for y, loc in enumerate(PYRAMID):
                if room in loc:
                    KNOWN_XY[int(room)] = (loc.index(room), y)
                    break
        return KNOWN_XY[room]

    def __getattr__(self, e):
        """Forward to gym environment."""
        return getattr(self.env, e)

    def get_ram(self):
        """Return the current RAM."""
        return ale_to_ram(self.env.unwrapped.ale)

    def reset(self, seed=None, return_info: bool = False) -> tuple[numpy.ndarray, dict[str, Any]]:
        """Reset the environment."""
        obs, info = self.env.reset()
        self.cur_lives = 5
        for _ in range(3):
            obs, *_, info = self.env.step(0)
        self.ram = self.get_ram()
        self.cur_score = 0
        self.cur_steps = 0
        self.pos = None
        self.pos = self.pos_from_obs(self.get_face_pixels(obs), obs)
        assert self.pos is not None
        if self.pos.room not in self.rooms:
            self.rooms[self.pos.room] = obs[50:].repeat(self._x_repeat, axis=1)
        if self.coords_obs:
            return self.get_coords(), info
        return obs, info

    def step(
        self, action
    ) -> (
        tuple[ndarray, SupportsFloat, bool, bool, dict[str, Any]]
        | tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]
    ):
        """Step the environment."""
        obs, reward, done, truncated, info = self.env.step(action)
        self.cur_score += reward
        self.cur_steps += 1
        self.ram = self.get_ram()

        face_pixels = self.get_face_pixels(obs)
        self.pos = self.pos_from_obs(face_pixels, obs)
        if self.check_death and (self.is_pixel_death(obs, face_pixels) or self.is_ram_death()):
            done = True
        elif self._death_room_8:  # pragma: no cover
            done = done or self.pos.room == 8

        if self.pos.room not in self.rooms:  # pragma: no cover
            self.rooms[self.pos.room] = obs[50:].repeat(self._x_repeat, axis=1)

        if self.coords_obs:
            return self.get_coords(), reward, done, truncated, info
        return obs, reward, done, truncated, info

    def pos_from_obs(self, face_pixels, obs) -> MontezumaPosLevel:
        """Extract the information of the position of Panama Joe."""
        face_pixels = [(y, x * self._x_repeat) for y, x in face_pixels]
        if len(face_pixels) == 0:
            assert self.pos is not None, "No face pixel and no previous pos"
            return self.pos  # Simply re-use the same position
        y, x = numpy.mean(face_pixels, axis=0)
        room = 1
        level = 0
        old_objects = ()
        if self.pos is not None:
            room = self.pos.room
            level = self.pos.level
            old_objects = self.pos.score
            direction_x = numpy.clip(int((self.pos.x - x) / 50), -1, 1)
            direction_y = numpy.clip(int((self.pos.y - y) / 50), -1, 1)
            if direction_x != 0 or direction_y != 0:  # pragma: no cover
                room_x, room_y = self.get_room_xy(self.pos.room)
                if room == 15 and room_y + direction_y >= len(PYRAMID):
                    room = 1
                    level += 1
                else:
                    assert direction_x == 0 or direction_y == 0, (
                        f"Room change in more than two directions : ({direction_y}, {direction_x})"
                    )
                    room = PYRAMID[room_y + direction_y][room_x + direction_x]
                    assert room != -1, f"Impossible room change: ({direction_y}, {direction_x})"

        score = self.cur_score
        if self.score_objects:  # TODO: detect objects from the frame!
            if not self.objects_from_pixels:
                score = self.ram[65]
                if self.only_keys:  # pragma: no cover
                    # These are the key bytes
                    score &= KEY_BITS
            else:
                score = self.get_objects_from_pixels(obs, room, old_objects)
        return MontezumaPosLevel(level, score, room, x, y)

    def get_objects_from_pixels(self, obs, room, old_objects):
        """Extract the position of the objects in the provided observation."""
        object_part = (obs[25:45, 55:110, 0] != 0).astype(numpy.uint8) * 255
        connected_components = cv2.connectedComponentsWithStats(object_part)
        pixel_areas = [e[-1] for e in connected_components[2]][1:]

        if self.objects_remember_rooms:
            cur_object = []
            old_objects = list(old_objects)
            for n_pixels in OBJECT_PIXELS:
                if n_pixels != 40 and self.only_keys:  # pragma: no cover
                    continue
                if n_pixels in pixel_areas:  # pragma: no cover
                    pixel_areas.remove(n_pixels)
                    orig_types = [e[0] for e in old_objects]
                    if n_pixels in orig_types:
                        idx = orig_types.index(n_pixels)
                        cur_object.append((n_pixels, old_objects[idx][1]))
                        old_objects.pop(idx)
                    else:
                        cur_object.append((n_pixels, room))

            return tuple(cur_object)

        cur_object = 0
        for i, n_pixels in enumerate(OBJECT_PIXELS):
            if n_pixels in pixel_areas:  # pragma: no cover
                pixel_areas.remove(n_pixels)
                cur_object |= 1 << i

        if self.only_keys:  # pragma: no cover
            # These are the key bytes
            cur_object &= KEY_BITS
        return cur_object

    def get_coords(self) -> numpy.ndarray:
        """Return an observation containing the position and the flattened screen of the game."""
        return numpy.array([self.pos.x, self.pos.y, self.pos.room, self.score_objects])

    def state_to_numpy(self) -> numpy.ndarray:
        """Return a numpy array containing the current state of the game."""
        state = self.unwrapped.clone_state()
        return numpy.array((state, None), dtype=object)

    def get_restore(self) -> tuple:
        """Return a tuple containing all the information needed to clone the state of the env."""
        return (
            self.state_to_numpy(),
            self.cur_score,
            self.cur_steps,
            self.pos,
            self.score_objects,
            self.cur_lives,
        )

    def restore(self, data) -> None:
        """Restore the state of the env from the provided tuple."""
        (
            full_state,
            score,
            steps,
            pos,
            self.score_objects,
            self.cur_lives,
        ) = data
        self.env.reset()
        self.unwrapped.restore_state(full_state)
        self.ram = self.get_ram()
        self.cur_score = score
        self.cur_steps = steps
        self.pos = pos

    def is_transition_screen(self, obs) -> bool:
        """Return True if the current observation corresponds to a transition between rooms."""
        obs = obs[50:, :, :]
        # The screen is a transition screen if it is all black or if its color is made
        # up only of black and (0, 28, 136), which is a color seen in the transition
        # screens between two levels.
        unprocessed_one = obs[:, :, 1]
        unprocessed_two = obs[:, :, 2]
        return (
            numpy.sum(obs[:, :, 0] == 0)
            + numpy.sum((unprocessed_one == 0) | (unprocessed_one == 28))
            + numpy.sum((unprocessed_two == 0) | (unprocessed_two == 136))
        ) == obs.size

    def get_face_pixels(self, obs) -> set:
        """Return the pixels containing the face of Paname Joe."""
        # TODO: double check that this color does not re-occur somewhere else
        # in the environment.
        return set(zip(*numpy.where(obs[50:, :, 0] == 228)))

    def is_pixel_death(self, obs, face_pixels):
        """Return a death signal extracted from the observation of the environment."""
        # There are no face pixels, and yet we are not in a transition screen. We
        # must be dead!
        if len(face_pixels) == 0:
            # All the screen except the bottom is black: this is not a death but a
            # room transition. Ignore.
            return not self.is_transition_screen(obs)  # pragma: no cover

        # We already checked for the presence of no face pixels, however,
        # sometimes we can die and still have face pixels. In those cases,
        # the face pixels will be DISCONNECTED.
        for pixel in face_pixels:
            for neighbor in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (
                    pixel[0] + neighbor[0],
                    pixel[1] + neighbor[1],
                ) in face_pixels:  # pragma: no cover
                    return False
        return True  # pragma: no cover

    def is_ram_death(self) -> bool:
        """Return a death signal extracted from the ram of the environment."""
        self.cur_lives = max(self.ram[58], self.cur_lives)
        return self.ram[55] != 0 or self.ram[58] < self.cur_lives

    def render(self, mode="human", **kwargs) -> None | numpy.ndarray:
        """Render the environment."""
        return self.env.render()


class MontezumaEnv(AtariEnv):
    """Plangym implementation of the MontezumaEnv environment optimized for planning."""

    AVAILABLE_OBS_TYPES = {"coords", "rgb", "grayscale", "ram", None}

    def __init__(
        self,
        name="PlanMontezuma-v0",
        frameskip: int = 1,
        episodic_life: bool = False,
        autoreset: bool = True,
        delay_setup: bool = False,
        remove_time_limit: bool = True,
        obs_type: str = "rgb",  # coords | ram | rgb | grayscale
        mode: int = 0,  # game mode, see Machado et al. 2018
        difficulty: int = 0,  # game difficulty, see Machado et al. 2018
        repeat_action_probability: float = 0.0,  # Sticky action probability
        full_action_space: bool = False,  # Use all actions
        render_mode: str | None = None,  # None | human | rgb_array
        possible_to_win: bool = True,
        wrappers: Iterable[wrap_callable] | None = None,
        array_state: bool = True,
        clone_seeds: bool = True,
        **kwargs,
    ):
        """Initialize a :class:`MontezumaEnv`."""
        super().__init__(
            name="MontezumaRevengeDeterministic-v4",
            frameskip=frameskip,
            autoreset=autoreset,
            episodic_life=episodic_life,
            clone_seeds=clone_seeds,
            delay_setup=delay_setup,
            remove_time_limit=remove_time_limit,
            obs_type=obs_type,
            mode=mode,
            difficulty=difficulty,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            render_mode=render_mode,
            possible_to_win=possible_to_win,
            wrappers=wrappers,
            array_state=array_state,
            **kwargs,
        )

    def _get_default_obs_type(self, name, obs_type) -> str:
        value = super()._get_default_obs_type(name, obs_type)
        if obs_type == "coords":
            value = obs_type
        return value

    def init_gym_env(self) -> CustomMontezuma:
        """Initialize the :class:`gum.Env`` instance that the current clas is wrapping."""
        kwargs = self._gym_env_kwargs
        kwargs["obs_type"] = self.obs_type
        return CustomMontezuma(**kwargs)

    def get_state(self) -> numpy.ndarray:
        """Recover the internal state of the simulation.

        If clone seed is False the environment will be stochastic.
        Cloning the full state ensures the environment is deterministic.
        """
        data = self.gym_env.get_restore()
        (
            full_state,
            score,
            steps,
            pos,
            score_objects,
            cur_lives,
        ) = data

        metadata = numpy.array(
            [
                float(score),
                float(steps),
                float(score_objects),
                float(cur_lives),
            ],
            dtype=float,
        )
        assert len(metadata) == 4
        posarray = numpy.array(pos.tuple, dtype=float)
        assert len(posarray) == 5
        return numpy.concatenate([full_state, metadata, posarray])

    def set_state(self, state: numpy.ndarray):
        """Set the internal state of the simulation.

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
        score, steps, score_objects, cur_lives = state[-9:-5].tolist()
        full_state = state[0]
        data = (
            full_state,
            score,
            steps,
            pos,
            bool(score_objects),
            int(cur_lives),
        )
        self.gym_env.restore(data)
