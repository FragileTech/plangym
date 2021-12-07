# This includes all the collection of wrappers from OpenAI gym and OpenAI baselines
# https://github.com/openai/gym/tree/master/gym/wrappers
# https://github.com/openai/baselines/blob/master/baselines/common
from gym import error
from gym.wrappers.monitor import Monitor

from plangym.wrappers.atari_wrappers import (
    AtariPreprocessing,
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    make_atari,
    MaxAndSkipEnv,
    NoopResetEnv,
    ScaledFloatFrame,
    WarpFrame,
    wrap_deepmind,
)
from plangym.wrappers.clip_action import ClipAction
from plangym.wrappers.filter_observation import FilterObservation
from plangym.wrappers.frame_stack import FrameStack, LazyFrames
from plangym.wrappers.gray_scale_observation import GrayScaleObservation
from plangym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from plangym.wrappers.rescale_action import RescaleAction
from plangym.wrappers.resize_observation import ResizeObservation
from plangym.wrappers.retro_wrappers import (
    AllowBacktracking,
    AppendTimeout,
    Downsample,
    make_retro,
    MovieRecord,
    PartialFrameStack,
    RewardScaler,
    Rgb2gray,
    SonicDiscretizer,
    StartDoingRandomActions,
    StochasticFrameSkip,
    wrap_deepmind_retro,
)
from plangym.wrappers.time_limit import TimeLimit
from plangym.wrappers.transform_observation import TransformObservation
from plangym.wrappers.transform_reward import TransformReward
