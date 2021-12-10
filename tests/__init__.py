try:
    import retro

    SKIP_RETRO_TESTS = False
except ImportError:
    SKIP_RETRO_TESTS = True

try:
    import ray

    SKIP_RAY_TESTS = False
except ImportError:
    SKIP_RAY_TESTS = True

try:
    from plangym.atari import AtariEnvironment

    AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)
    SKIP_ATARI_TESTS = False
except Exception:
    SKIP_ATARI_TESTS = True

try:
    from plangym.dm_control import DMControlEnv

    DMControlEnv(name="walker-run", frameskip=3)
    SKIP_DM_CONTROL_TESTS = False
except ImportError:
    SKIP_DM_CONTROL_TESTS = True


try:
    import Box2D

    SKIP_BOX2D_TESTS = False
except ImportError:
    SKIP_BOX2D_TESTS = True
