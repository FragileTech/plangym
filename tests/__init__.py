try:
    from plangym.atari import AtariEnvironment

    AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)
    SKIP_ATARI_TESTS = False
except Exception:
    SKIP_ATARI_TESTS = True
