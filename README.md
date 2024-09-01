## Welcome to Plangym

[![Documentation Status](https://readthedocs.org/projects/plangym/badge/?version=latest)](https://plangym.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://codecov.io/github/FragileTech/plangym/coverage.svg)](https://codecov.io/github/FragileTech/plangym)
[![PyPI package](https://badgen.net/pypi/v/plangym)](https://pypi.org/project/plangym/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Plangym is an open source Python library for developing and comparing planning algorithms by providing a 
standard API to communicate between algorithms and environments, as well as a standard set of environments 
compliant with that API.

Given that OpenAI's `gym` has become the de-facto standard in the research community, `plangym`'s API 
is designed to be as similar as possible to `gym`'s API while allowing to modify the environment state.

Furthermore, it provides additional functionality for stepping the environments in parallel, delayed environment
initialization for dealing with environments that are difficult to serialize, compatibility with `gym.Wrappers`, 
and more.

## Supported environments
Plangym currently supports all the following environments:

* OpenAI gym classic control environments
* OpenAI gym Box2D environments
* OpenAI gym Atari 2600 environments
* Deepmind's dm_control environments
* Stable-retro environments

## Getting started

### Stepping an environment
```python
import plangym
env = plangym.make(name="CartPole-v0")
state, obs = env.reset()

state = state.copy()
action = env.action_space.sample()

data = env.step(state=state, action=action)
new_state, observ, reward, end, truncated, info = data
```


### Stepping a batch of states and actions
```python
import plangym
env = plangym.make(name="CartPole-v0")
state, obs = env.reset()

states = [state.copy() for _ in range(10)]
actions = [env.action_space.sample() for _ in range(10)]

data = env.step_batch(states=states, actions=actions)
new_states, observs, rewards, ends, truncateds, infos = data
```


### Using parallel steps

```python
import plangym
env = plangym.make(name="MsPacman-v0", n_workers=2)

state, obs = env.reset()

states = [state.copy() for _ in range(10)]
actions = [env.action_space.sample() for _ in range(10)]

data =  env.step_batch(states=states, actions=actions)
new_states, observs, rewards, ends, truncateds, infos = data
```

## Installation 
TODO: Meanwhile take a look at how we set up the repository in `.github/workflows/push.yaml`.

## License
Plangym is released under the [MIT](LICENSE) license.

## Contributing

Contributions are very welcome! Please check the [contributing guidelines](CONTRIBUTING.md) before opening a pull request.

If you have any suggestions for improvement, or you want to report a bug please open 
an [issue](https://github.com/FragileTech/plangym/issues).


# Installing nes-py

#### Step 1: Install necessary development tools and libraries
sudo apt-get update
sudo apt-get install build-essential clang
sudo apt-get install libstdc++-10-dev

#### Step 2: Verify the compiler and include paths
#### Ensure you are using g++ instead of clang++ if clang++ is not properly configured
export CXX=g++
export CC=gcc

# Rebuild the project
rye install nes-py --git=https://github.com/FragileTech/nes-py