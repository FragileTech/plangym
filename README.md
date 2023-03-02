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
new_state, observ, reward, end, info = data
```


### Stepping a batch of states and actions
```python
import plangym
env = plangym.make(name="CartPole-v0")
state, obs = env.reset()

states = [state.copy() for _ in range(10)]
actions = [env.action_space.sample() for _ in range(10)]

data = env.step_batch(states=states, actions=actions)
new_states, observs, rewards, ends, infos = data
```


### Using parallel steps

```python
import plangym
env = plangym.make(name="MsPacman-v0", n_workers=2)

state, obs = env.reset()

states = [state.copy() for _ in range(10)]
actions = [env.action_space.sample() for _ in range(10)]

data =  env.step_batch(states=states, actions=actions)
new_states, observs, rewards, ends, infos = data
```

## Installation 
Plangym is tested on Ubuntu 20.04 and Ubuntu 21.04 for python versions 3.7 and 3.8.

Installing it with Python 3.6 will break AtariEnv, and RetroEnv does not support 
python 3.9 yet.

### Installing from Pip
Assuming that the environment libraries that you want to use are already installed, you can 
install plangym from pip running:
```bash
pip3 install plangym
```

### Installing from source
If you also want to install the environment libraries, first clone the repository:

```bash
git clone git@github.com:FragileTech/plangym.git
cd plangym
```

Install the system dependencies by running
```bash
sudo apt-get install -y --no-install-suggests --no-install-recommends libglfw3 libglew-dev libgl1-mesa-glx libosmesa6 xvfb swig
```

To install MuJoCo, run:
```bash
make install-mujoco
```

Finally, install the project requirements and plangym.
```bash
pip install -r requirements.txt
pip install .
```

## Roadmap

This is a summary of the incoming improvements to the project:
- **Improved documentation**:
  * Adding specific tutorials for all the different types of supported environments.
  * Adding a developer guide section for incorporating new environments to plangym.
  * Improving the library docstrings with more examples and detailed information.
- **Better gym integration**:
  * Registering all of plangym environments in gym under a namespace.
  * Offering more control over how the states are passed to `step`, `reset` and `step_batch`.
  * Allowing to return the states inside the info dictionary.
- **Adding new environments to plangym, such as**:
  * Gym mujoco
  * Gym robotics
  * [Gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- **Support for rendering in notebooks that are running on headless machines**.

## License
Plangym is released under the [MIT](LICENSE) license.

## Contributing

Contributions are very welcome! Please check the [contributing guidelines](CONTRIBUTING.md) before opening a pull request.

If you have any suggestions for improvement, or you want to report a bug please open 
an [issue](https://github.com/FragileTech/plangym/issues).
