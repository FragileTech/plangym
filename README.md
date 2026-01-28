# Plangym

[![Documentation Status](https://readthedocs.org/projects/plangym/badge/?version=latest)](https://plangym.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://codecov.io/github/FragileTech/plangym/coverage.svg)](https://codecov.io/github/FragileTech/plangym)
[![PyPI package](https://badgen.net/pypi/v/plangym)](https://pypi.org/project/plangym/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Plangym** is a Python library that extends [Gymnasium](https://gymnasium.farama.org/) environments for planning algorithms. It provides the ability to get and set complete environment state, enabling deterministic rollouts from arbitrary states—critical for planning algorithms that need to branch execution.

### Key Features

- **State manipulation**: `get_state()` and `set_state()` for full environment state control
- **Batch stepping**: Execute multiple state-action pairs in a single call
- **Parallel execution**: Built-in multiprocessing and Ray support for distributed rollouts
- **Gymnasium compatible**: Works with `gym.Wrappers` and standard Gym API
- **Delayed initialization**: Serialize environments before setup for distributed workers

---

## Table of Contents

- [Supported Environments](#supported-environments)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Developer Guide](#developer-guide)
- [Local CI with act](#local-ci-with-act)
- [Architecture](#architecture)
- [License](#license)
- [Contributing](#contributing)

---

## Supported Environments

| Environment Type | Package | Description |
|-----------------|---------|-------------|
| **Classic Control** | `gymnasium` | CartPole, Pendulum, MountainCar, etc. |
| **Box2D** | `gymnasium[box2d]` | LunarLander, BipedalWalker, CarRacing |
| **Atari** | `ale-py` | Atari 2600 games via Arcade Learning Environment |
| **dm_control** | `dm-control` | DeepMind Control Suite with MuJoCo physics |
| **MuJoCo** | `mujoco` | MuJoCo physics environments |
| **Retro** | `stable-retro` | Classic console games (Genesis, SNES, etc.) |
| **NES** | `nes-py` | NES games including Super Mario Bros |

---

## Requirements

### Python Version

- **Python 3.10 or higher**

### System Dependencies

<details>
<summary><strong>Ubuntu / Debian</strong></summary>

```bash
sudo apt-get update
sudo apt-get install -y xvfb libglu1-mesa x11-utils
```

For NES environments (nes-py):
```bash
sudo apt-get install -y build-essential clang libstdc++-10-dev
```

</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install --cask xquartz
brew install swig libzip

# Create X11 socket directory if needed
if [ ! -d /tmp/.X11-unix ]; then
    sudo mkdir /tmp/.X11-unix
    sudo chmod 1777 /tmp/.X11-unix
    sudo chown root /tmp/.X11-unix
fi
```

</details>

<details>
<summary><strong>WSL2 (Windows)</strong></summary>

```bash
sudo apt-get update
sudo apt-get install -y xvfb libglu1-mesa x11-utils
```

For GUI rendering, install an X server on Windows (e.g., VcXsrv) or use headless mode.

</details>

---

## Installation

### Quick Install

```bash
# Using pip
pip install plangym

# Using uv
uv add plangym
```

### Install with Optional Extras

Plangym provides optional extras for different environment types:

| Extra | Description | Includes |
|-------|-------------|----------|
| `atari` | Atari 2600 games | `ale-py`, `gymnasium[atari]` |
| `nes` | NES / Super Mario | `nes-py`, `gym-super-mario-bros` |
| `classic-control` | Classic control envs | `gymnasium[classic_control]`, `pygame` |
| `dm_control` | DeepMind Control Suite | `mujoco`, `dm-control` |
| `retro` | Retro console games | `stable-retro` |
| `box_2d` | Box2D physics | `box2d-py` |
| `ray` | Distributed computing | `ray` |
| `jupyter` | Notebook support | `jupyterlab` |

```bash
# Install specific extras
pip install "plangym[atari,dm_control]"

# Install all environment extras
pip install "plangym[atari,nes,classic-control,dm_control,retro,box_2d,ray]"
```

### Development Installation

```bash
git clone https://github.com/FragileTech/plangym.git
cd plangym
uv sync --all-extras
```

### ROM Installation

For Retro environments, you need to import ROM files:

```bash
# Retro ROMs (requires ROM files)
python -m plangym.scripts.import_retro_roms
```

Note: Atari ROMs are now bundled with `ale-py` >= 0.9, so no additional installation is needed for Atari environments.

---

## Quick Start

### Basic Environment Stepping

```python
import plangym

env = plangym.make(name="CartPole-v1")
state, obs, info = env.reset()

# Save state for later
saved_state = state.copy()

# Take a step
action = env.action_space.sample()
new_state, obs, reward, terminated, truncated, info = env.step(state=state, action=action)

# Restore to saved state and try a different action
different_action = env.action_space.sample()
new_state2, obs2, reward2, _, _, _ = env.step(state=saved_state, action=different_action)
```

### Batch Stepping

Execute multiple state-action pairs efficiently:

```python
import plangym

env = plangym.make(name="CartPole-v1")
state, obs, info = env.reset()

# Create batch of states and actions
states = [state.copy() for _ in range(10)]
actions = [env.action_space.sample() for _ in range(10)]

# Step all at once
new_states, observations, rewards, terminateds, truncateds, infos = env.step_batch(
    states=states,
    actions=actions
)
```

### Parallel Execution

Use multiple workers for faster rollouts:

```python
import plangym

# Create environment with 4 parallel workers
env = plangym.make(name="ALE/MsPacman-v5", n_workers=4)

state, obs, info = env.reset()

states = [state.copy() for _ in range(100)]
actions = [env.action_space.sample() for _ in range(100)]

# Steps are distributed across workers
new_states, observations, rewards, terminateds, truncateds, infos = env.step_batch(
    states=states,
    actions=actions
)
```

---

## Developer Guide

### Development Setup

```bash
git clone https://github.com/FragileTech/plangym.git
cd plangym
uv sync --all-extras
```

### Code Style

Plangym uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Auto-fix and format code
make style

# Check code style (no modifications)
make check
```

### Running Tests

```bash
# Run full test suite
make test

# Run tests in parallel (default: 2 workers)
make test-parallel

# Run tests with custom worker count
n=4 make test-parallel

# Run classic control tests (single-threaded)
make test-singlecore

# Run doctests
make test-doctest
```

**Running individual test files:**

```bash
# dm_control tests (requires MUJOCO_GL for headless rendering)
MUJOCO_GL=egl uv run pytest tests/control/test_dm_control.py -s

# Specific test
uv run pytest tests/test_core.py::TestCoreEnv::test_step -v
```

**Environment Variables:**

| Variable | Description |
|----------|-------------|
| `MUJOCO_GL=egl` | Headless MuJoCo rendering |
| `PYVIRTUALDISPLAY_DISPLAYFD=0` | Virtual display for rendering tests |
| `SKIP_CLASSIC_CONTROL=1` | Skip classic control in parallel runs |
| `SKIP_RENDER=True` | Skip rendering tests |
| `n=2` | Number of parallel test workers |

### Code Coverage

```bash
# Run all coverage targets
make codecov

# Individual coverage targets
make codecov-parallel    # Parallel tests
make codecov-singlecore  # Single-core tests
```

### Building Documentation

```bash
# Build Sphinx documentation
make build-docs

# Serve documentation locally
make serve-docs
```

### Docker

```bash
# Build Docker image
make docker-build

# Run interactive shell in container
make docker-shell

# Run tests in Docker
make docker-test

# Run Jupyter notebook server
make docker-notebook
```

---

## Local CI with act

[act](https://github.com/nektos/act) allows you to run GitHub Actions workflows locally for debugging.

### Prerequisites

1. **Docker** installed and running
   - For WSL2: Enable Docker Desktop WSL Integration in Settings → Resources → WSL Integration

2. **act** installed:
   ```bash
   # macOS
   brew install act

   # Linux (using Go)
   go install github.com/nektos/act@latest

   # Or download from GitHub releases
   ```

### Configuration

Plangym includes pre-configured act settings:

- **`.actrc`** - Default act configuration
- **`.secrets`** - Local secrets file (gitignored)

### Running Workflows Locally

```bash
# List all available jobs
act -l

# Run specific jobs
act -j style-check        # Lint check
act -j pytest             # Run tests
act -j build-test-package # Build and test package

# Dry run (see what would execute)
act -n

# Run with verbose output
act -j style-check -v
```

### Secrets Setup

Edit `.secrets` to add your credentials for full CI functionality:

```bash
# .secrets file format
ROM_PASSWORD=your_rom_password
CODECOV_TOKEN=your_codecov_token
TEST_PYPI_PASS=your_test_pypi_token
BOT_AUTH_TOKEN=your_github_bot_token
PYPI_PASS=your_pypi_token
```

> **Note:** The `.secrets` file is gitignored and should never be committed.

### Troubleshooting act

<details>
<summary><strong>Docker not found in WSL2</strong></summary>

Enable WSL integration in Docker Desktop:
1. Open Docker Desktop
2. Go to Settings → Resources → WSL Integration
3. Enable integration for your WSL distro
4. Restart Docker Desktop

</details>

<details>
<summary><strong>Job runs but fails on specific actions</strong></summary>

Some GitHub Actions may not work perfectly with act. Common issues:
- `actions/cache` - May need `--reuse` flag
- Platform-specific steps - act only runs Linux containers
- Service containers - May require additional configuration

</details>

---

## Architecture

```
plangym/
├── core.py           # PlanEnv, PlangymEnv base classes
├── registry.py       # make() factory function
├── control/          # Physics environments
│   ├── classic_control.py
│   ├── dm_control.py
│   ├── mujoco.py
│   └── box2d.py
├── videogames/       # Emulator environments
│   ├── atari.py
│   ├── retro.py
│   └── nes.py
└── vectorization/    # Parallel execution
    ├── env.py        # VectorizedEnv base
    ├── parallel.py   # Multiprocessing
    └── ray.py        # Ray distributed
```

### Core Classes

| Class | Description |
|-------|-------------|
| `PlanEnv` | Abstract base defining `get_state()`, `set_state()`, `step()` interface |
| `PlangymEnv` | Wraps Gymnasium environments with state manipulation |
| `VectorizedEnv` | Base for parallel execution backends |
| `ParallelEnv` | Multiprocessing-based parallel stepping |
| `RayEnv` | Ray-based distributed stepping |

### Entry Point

```python
import plangym

# The make() function routes to the correct environment class
env = plangym.make(
    name="CartPole-v1",    # Environment name
    n_workers=4,            # Parallel workers (optional)
    obs_type="rgb",         # Observation type: coords, rgb, grayscale
    delay_setup=True,       # Defer initialization for serialization
)
```

---

## License

Plangym is released under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

**Quick contribution checklist:**

1. Run `make check` to verify code style
2. Run `make test` to ensure tests pass
3. Add tests for new functionality
4. Update documentation as needed

For bug reports and feature requests, please open an [issue](https://github.com/FragileTech/plangym/issues).
