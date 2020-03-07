"""Various environments for plangym."""
from plangym.dm_control import DMControlEnv, ParallelDMControl
from plangym.env import AtariEnvironment, ParallelEnvironment
from plangym.minimal import ClassicControl, MinimalPacman, MinimalPong
