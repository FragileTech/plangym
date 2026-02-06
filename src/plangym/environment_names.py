"""Lists of available environments."""

CLASSIC_CONTROL = [
    "CartPole-v0",
    "CartPole-v1",
    "MountainCarContinuous-v0",
    "MountainCar-v0",
    "Pendulum-v1",
    "Acrobot-v1",
]

BOX_2D = [
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "CarRacing-v3",
    "FastLunarLander-v0",
]

MUJOCO = [
    "Ant-v4",
    "Ant-v5",
    "HalfCheetah-v4",
    "HalfCheetah-v5",
    "Hopper-v4",
    "Hopper-v5",
    "Humanoid-v4",
    "Humanoid-v5",
    "HumanoidStandup-v4",
    "HumanoidStandup-v5",
    "InvertedDoublePendulum-v4",
    "InvertedDoublePendulum-v5",
    "InvertedPendulum-v4",
    "InvertedPendulum-v5",
    "Pusher-v4",
    "Pusher-v5",
    "Reacher-v4",
    "Reacher-v5",
    "Swimmer-v4",
    "Swimmer-v5",
    "Walker2d-v4",
    "Walker2d-v5",
]

_RETRO = None
_DM_CONTROL = None


def _load_retro():
    global _RETRO
    if _RETRO is None:
        try:
            import retro.data

            _RETRO = retro.data.list_games()
        except Exception:  # pragma: no cover
            _RETRO = []
    return _RETRO


def _load_dm_control():
    global _DM_CONTROL
    if _DM_CONTROL is None:
        try:
            from dm_control import suite

            _DM_CONTROL = list(suite.ALL_TASKS)
        except (ImportError, OSError, AttributeError):  # pragma: no cover
            _DM_CONTROL = []
    return _DM_CONTROL


def __getattr__(name: str):
    if name == "RETRO":
        return _load_retro()
    if name == "DM_CONTROL":
        return _load_dm_control()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


ATARI = [
    "ALE/Boxing-v5",
    "ALE/RoadRunner-v5",
    "ALE/Pitfall2-v5",
    "ALE/Amidar-v5",
    "ALE/Zaxxon-v5",
    "ALE/MsPacman-v5",
    "ALE/Breakout-v5",
    "ALE/FishingDerby-v5",
    "ALE/Adventure-v5",
    "ALE/BankHeist-v5",
    "ALE/Frostbite-v5",
    "ALE/ChopperCommand-v5",
    "ALE/Pitfall-v5",
    "ALE/YarsRevenge-v5",
    "ALE/KungFuMaster-v5",
    "ALE/Alien-v5",
    "ALE/BattleZone-v5",
    "ALE/Pooyan-v5",
    "ALE/Phoenix-v5",
    "ALE/CrazyClimber-v5",
    "ALE/Assault-v5",
    "ALE/Hero-v5",
    "ALE/Venture-v5",
    "ALE/TimePilot-v5",
    "ALE/Defender-v5",
    "ALE/Bowling-v5",
    "ALE/NameThisGame-v5",
    "ALE/DoubleDunk-v5",
    "ALE/Jamesbond-v5",
    "ALE/BeamRider-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/AirRaid-v5",
    "ALE/DemonAttack-v5",
    "ALE/Atlantis2-v5",
    "ALE/Kangaroo-v5",
    "ALE/Tutankham-v5",
    "ALE/Freeway-v5",
    "ALE/Gravitar-v5",
    "ALE/IceHockey-v5",
    "ALE/ElevatorAction-v5",
    "ALE/VideoPinball-v5",
    "ALE/Centipede-v5",
    "ALE/Berzerk-v5",
    "ALE/Riverraid-v5",
    "ALE/Tennis-v5",
    "ALE/Enduro-v5",
    "ALE/Qbert-v5",
    "ALE/UpNDown-v5",
    "ALE/PrivateEye-v5",
    "ALE/Asteroids-v5",
    "ALE/Carnival-v5",
    "ALE/StarGunner-v5",
    "ALE/JourneyEscape-v5",
    "ALE/Asterix-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/Gopher-v5",
    "ALE/Skiing-v5",
    "ALE/Atlantis-v5",
    "ALE/Solaris-v5",
    "ALE/Seaquest-v5",
    "ALE/WizardOfWor-v5",
    "ALE/Krull-v5",
    "ALE/Pong-v5",
    "ALE/Robotank-v5",
]
