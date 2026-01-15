# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Plangym is a Python library that extends Gymnasium (OpenAI Gym) environments for planning algorithms. The key differentiator is the ability to get/set complete environment state, enabling deterministic rollouts from arbitrary states - critical for planning algorithms that need to branch execution.

## Common Commands

### Development
```bash
make style          # Format code with ruff (auto-fix)
make check          # Check code style without modifying
```

### Testing
```bash
# Full test suite
make test

# Individual test targets
make test-parallel      # Run parallel tests (uses pytest-xdist with n workers)
make test-singlecore    # Run classic control tests single-threaded
make test-doctest       # Run doctests in source files

# Run a single test file
MUJOCO_GL=egl PYVIRTUALDISPLAY_DISPLAYFD=0 uv run pytest tests/control/test_dm_control.py -s

# Run a specific test
uv run pytest tests/test_core.py::TestCoreEnv::test_step -v
```

### Documentation
```bash
make build-docs     # Build sphinx documentation
make serve-docs     # Serve docs locally
```

### Environment Variables
- `MUJOCO_GL=egl` - Required for MuJoCo rendering in headless environments
- `PYVIRTUALDISPLAY_DISPLAYFD=0` - Virtual display for rendering tests
- `SKIP_CLASSIC_CONTROL=1` - Skip classic control tests in parallel runs
- `n=2` - Number of parallel workers (default 2)

## Architecture

### Core Abstraction Hierarchy

```
PlanEnv (Abstract Base - src/plangym/core.py)
└── PlangymEnv (Gym Wrapper)
    ├── ClassicControl, Box2DEnv, DMControlEnv (control/)
    └── VideogameEnv (videogames/)
        ├── AtariEnv, RetroEnv, MarioEnv
        └── MontezumaEnv
```

### Key Classes

- **`PlanEnv`** (`core.py`): Abstract base defining the interface. Key methods: `get_state()`, `set_state()`, `apply_action()`, `apply_reset()`. Subclasses must implement these to enable planning.

- **`PlangymEnv`** (`core.py`): Wraps Gymnasium environments. Handles observation types (`coords`, `rgb`, `grayscale`), wrapper composition, and time limit removal.

- **`VectorizedEnv`** (`vectorization/env.py`): Base for parallel execution. Implementations: `ParallelEnv` (multiprocessing), `RayEnv` (distributed).

### Entry Point

The `make()` function in `registry.py` is the main factory. It routes to the correct environment class based on the environment name:
```python
import plangym
env = plangym.make(name="CartPole-v1")  # Classic control
env = plangym.make(name="ALE/MsPacman-v5", n_workers=4)  # Parallel Atari
```

### Design Patterns

1. **Delayed Setup**: `delay_setup=True` defers initialization, enabling serialization before setup (for distributed workers).

2. **Multi-step Execution**: `dt` parameter applies same action multiple times; combined with `frameskip` for total steps.

3. **Post-processing Hooks**: Override `process_obs()`, `process_reward()`, `process_terminal()`, `process_info()` in subclasses for custom transformations.

### Test Framework

Shared test base classes in `src/plangym/api_tests.py`:
- `TestPlanEnv`: Tests state management
- `TestPlangymEnv`: Tests Gym-wrapped environments
- `generate_test_cases()`: Parameterized tests across obs_types, render_modes, workers

## Source Layout

- `src/plangym/core.py` - Base classes (`PlanEnv`, `PlangymEnv`)
- `src/plangym/registry.py` - `make()` factory function
- `src/plangym/control/` - Physics-based environments (classic control, dm_control, mujoco)
- `src/plangym/videogames/` - Emulator-based environments (Atari, Retro, Mario)
- `src/plangym/vectorization/` - Parallel execution (`ParallelEnv`, `RayEnv`)
