from plangym.env import AtariEnvironment


def test_env():
    env = AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)
    state, obs = env.reset()

    states = [state.copy() for _ in range(10)]
    actions = [env.action_space.sample() for _ in range(10)]

    data = env.step_batch(states=states, actions=actions)
    assert data is not None
