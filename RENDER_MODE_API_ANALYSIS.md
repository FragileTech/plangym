# Render Mode API Analysis: Plangym vs Gymnasium Standards

This document analyzes two approaches for handling `render_mode` in plangym:
- **Option A**: Implement dynamic render mode switching (plangym-style)
- **Option B**: Conform to Gymnasium's immutable render_mode standard

## Executive Summary

| Aspect | Option A (Dynamic) | Option B (Gymnasium-Compliant) |
|--------|-------------------|-------------------------------|
| Complexity | High | Low |
| Gymnasium compatibility | Breaks standard | Full compliance |
| Breaking changes | Potentially many | Minimal (bug fix) |
| Maintenance burden | High | Low |
| **Recommendation** | | **Recommended** |

**Recommendation**: Option B. The current code has a bug that ignores the `render_mode` parameter. Fixing this bug to properly respect the parameter aligns with Gymnasium standards and requires minimal changes.

---

## 1. Current State Analysis

### The Bug at `src/plangym/core.py:537-539`

```python
def __init__(
    self,
    name: str,
    frameskip: int = 1,
    autoreset: bool = True,
    wrappers: Iterable[wrap_callable] | None = None,
    delay_setup: bool = False,
    remove_time_limit: bool = True,
    render_mode: str | None = "rgb_array",  # Parameter accepts user input
    episodic_life=False,
    obs_type=None,
    return_image=False,
    **kwargs,
):
    """..."""
    render_mode = "rgb_array"  # LINE 537: BUG - Overwrites the parameter!
    kwargs["render_mode"] = kwargs.get("render_mode", render_mode)
    self._render_mode = render_mode
```

**Problem**: Line 537 unconditionally overwrites the `render_mode` parameter to `"rgb_array"`, making the parameter useless. Users cannot set `render_mode=None` for faster training or `render_mode="human"` for visualization.

### Current render_mode Property (Read-Only)

```python
# core.py:621-624
@property
def render_mode(self) -> None | str:
    """Return how the game will be rendered. Values: None | human | rgb_array."""
    return self._render_mode
```

There is no setter, so render_mode cannot be changed after initialization.

---

## 2. Gymnasium Standard for render_mode

### Official Behavior (Gymnasium v0.29+)

From the [Gymnasium documentation](https://gymnasium.farama.org/api/env/):

1. **render_mode is immutable after `__init__`**
2. Must be specified at environment creation via `gymnasium.make()`
3. Cannot be changed dynamically during the environment's lifetime
4. Valid modes: `None`, `"human"`, `"rgb_array"`, `"ansi"`

### Why Gymnasium Made This Design Choice

1. **Performance**: `render_mode=None` allows environments to skip rendering entirely
2. **Initialization**: Rendering objects (OpenGL contexts, windows) are created once in `__init__`
3. **Consistency**: Prevents bugs from mode switching mid-episode
4. **Simplicity**: No need for boilerplate mode-switching code

### Usage Pattern in Gymnasium

```python
# For training (fast, no rendering)
training_env = gymnasium.make("CartPole-v1", render_mode=None)

# For evaluation (visual feedback)
eval_env = gymnasium.make("CartPole-v1", render_mode="human")

# For recording/analysis (get pixel arrays)
recording_env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
```

---

## 3. Option A: Implement Dynamic Render Mode Switching

This option would allow changing `render_mode` after environment initialization.

### Implementation Requirements

#### 3.1 Add render_mode Setter

```python
# In PlangymEnv class (core.py)

@render_mode.setter
def render_mode(self, mode: str | None):
    """Dynamically change the render mode.

    WARNING: This requires recreating the underlying gym environment,
    which may be expensive and reset state.
    """
    if mode not in self.AVAILABLE_RENDER_MODES:
        raise ValueError(f"Invalid render_mode: {mode}. Must be one of {self.AVAILABLE_RENDER_MODES}")

    if mode == self._render_mode:
        return  # No change needed

    old_mode = self._render_mode
    self._render_mode = mode
    self._gym_env_kwargs["render_mode"] = mode

    if self._gym_env is not None:
        self._recreate_gym_env_with_new_mode(old_mode, mode)

def _recreate_gym_env_with_new_mode(self, old_mode: str | None, new_mode: str | None):
    """Recreate the gym environment with a new render mode."""
    # Save current state if possible
    try:
        state = self.get_state()
        has_state = True
    except Exception:
        has_state = False

    # Close old environment
    if hasattr(self._gym_env, "close"):
        self._gym_env.close()

    # Create new environment with new mode
    self._gym_env = None  # Clear reference
    self.setup()  # Reinitialize

    # Restore state if we had one
    if has_state:
        try:
            self.set_state(state)
        except Exception:
            # State restoration failed, environment is reset
            pass
```

#### 3.2 Handle Wrappers

Wrappers must be reapplied after recreation:

```python
def _recreate_gym_env_with_new_mode(self, old_mode, new_mode):
    # ... save state ...

    # Store wrapper chain
    wrappers_to_reapply = self._wrappers

    # Close and recreate
    self._gym_env.close()
    self._gym_env = self.init_gym_env()

    # Reapply wrappers
    if wrappers_to_reapply:
        self.apply_wrappers(wrappers_to_reapply)

    # ... restore state ...
```

#### 3.3 Handle Vectorized Environments

`ParallelEnv` and `RayEnv` spawn worker processes with copies of the environment. Dynamic mode changes must propagate to workers:

```python
# In VectorizedEnv (vectorization/env.py)

def set_render_mode(self, mode: str | None):
    """Change render mode across all workers."""
    self._render_mode = mode
    # Send message to all workers to recreate their environments
    for worker in self._workers:
        worker.send(("set_render_mode", mode))
```

#### 3.4 Handle DMControlEnv

DMControlEnv doesn't use gymnasium, so it needs separate handling:

```python
# In DMControlEnv (control/dm_control.py)

@render_mode.setter
def render_mode(self, mode: str | None):
    """DMControlEnv can change render mode without recreation."""
    self._render_mode = mode
    # dm_control environments don't need recreation for mode change
```

### Risks and Downsides of Option A

1. **Breaks Gymnasium Contract**: Other tools expecting immutable render_mode may break
2. **State Loss Risk**: Environment recreation may lose internal state that `get_state()` doesn't capture
3. **Performance Overhead**: Recreation is expensive (reload models, reinitialize physics)
4. **Worker Synchronization**: Complex coordination with parallel workers
5. **Wrapper State**: Some wrappers maintain internal state that would be lost
6. **Testing Burden**: Need extensive tests for all mode transition combinations
7. **Maintenance Cost**: Ongoing complexity in all environment subclasses

### Files to Modify for Option A

| File | Changes |
|------|---------|
| `src/plangym/core.py` | Add setter, `_recreate_gym_env_with_new_mode()` |
| `src/plangym/vectorization/env.py` | Add `set_render_mode()`, worker communication |
| `src/plangym/vectorization/parallel.py` | Handle mode change messages in workers |
| `src/plangym/vectorization/ray.py` | Handle mode change in Ray actors |
| `src/plangym/control/dm_control.py` | Override setter (no recreation needed) |
| `tests/test_core.py` | Add dynamic mode switching tests |
| `tests/vectorization/` | Add parallel mode switching tests |

---

## 4. Option B: Conform to Gymnasium Standard (Recommended)

This option fixes the bug to properly respect the `render_mode` parameter while maintaining Gymnasium's immutable-after-init semantics.

### Implementation Requirements

#### 4.1 Fix the Bug in `__init__`

```python
# In PlangymEnv.__init__ (core.py:537-539)

# BEFORE (buggy):
render_mode = "rgb_array"  # DELETE THIS LINE
kwargs["render_mode"] = kwargs.get("render_mode", render_mode)
self._render_mode = render_mode

# AFTER (fixed):
self._render_mode = render_mode  # Use the parameter as-is
kwargs["render_mode"] = kwargs.get("render_mode", render_mode)
```

**Complete fixed `__init__` signature section:**

```python
def __init__(
    self,
    name: str,
    frameskip: int = 1,
    autoreset: bool = True,
    wrappers: Iterable[wrap_callable] | None = None,
    delay_setup: bool = False,
    remove_time_limit: bool = True,
    render_mode: str | None = "rgb_array",  # Default remains rgb_array for backward compat
    episodic_life=False,
    obs_type=None,
    return_image=False,
    **kwargs,
):
    # Validate render_mode
    if render_mode is not None and render_mode not in self.AVAILABLE_RENDER_MODES:
        raise ValueError(
            f"Invalid render_mode: {render_mode}. "
            f"Must be one of {self.AVAILABLE_RENDER_MODES}"
        )

    self._render_mode = render_mode  # Properly store the parameter
    kwargs["render_mode"] = kwargs.get("render_mode", render_mode)
    # ... rest of init ...
```

#### 4.2 Handle `get_image()` When render_mode is None

```python
# In PlangymEnv.get_image() (core.py:700-715)

def get_image(self) -> numpy.ndarray:
    """Return a numpy array containing the rendered view of the environment."""
    if self.render_mode is None:
        raise RuntimeError(
            "Cannot get image when render_mode=None. "
            "Create the environment with render_mode='rgb_array' to enable image capture."
        )

    if hasattr(self.gym_env, "render"):
        img = self.gym_env.render()
        if img is None and self.render_mode == "rgb_array":
            raise ValueError(f"Rendering rgb_array but got None: {self}")
        return img
    raise NotImplementedError()
```

#### 4.3 Handle `return_image=True` with `render_mode=None`

```python
# In PlangymEnv.__init__, add validation:

if return_image and render_mode is None:
    raise ValueError(
        "return_image=True requires render_mode='rgb_array', "
        "but render_mode=None was specified."
    )
```

#### 4.4 Update Tests

```python
# In tests/test_core.py or api_tests.py

def test_render_mode_none_fast_training():
    """Verify render_mode=None works for fast training."""
    env = plangym.make("CartPole-v1", render_mode=None)
    state, obs, info = env.reset()
    assert "rgb" not in info  # No image when render_mode=None
    env.close()

def test_render_mode_rgb_array():
    """Verify render_mode='rgb_array' returns images."""
    env = plangym.make("CartPole-v1", render_mode="rgb_array", return_image=True)
    state, obs, info = env.reset()
    assert "rgb" in info
    assert isinstance(info["rgb"], numpy.ndarray)
    env.close()

def test_render_mode_respected():
    """Verify the render_mode parameter is actually used."""
    env = plangym.make("CartPole-v1", render_mode=None)
    assert env.render_mode is None

    env2 = plangym.make("CartPole-v1", render_mode="rgb_array")
    assert env2.render_mode == "rgb_array"
```

### Files to Modify for Option B

| File | Changes |
|------|---------|
| `src/plangym/core.py` | Remove line 537, add validation |
| `tests/test_core.py` | Add render_mode validation tests |

### Benefits of Option B

1. **Simple Fix**: Just remove one line and add validation
2. **Gymnasium Compliance**: Works correctly with all gymnasium tooling
3. **Performance**: Users can now use `render_mode=None` for faster training
4. **Backward Compatible**: Default remains `"rgb_array"` for existing code
5. **Low Risk**: Minimal code changes, easy to review and test

---

## 5. Comparison Summary

| Criteria | Option A (Dynamic) | Option B (Gymnasium) |
|----------|-------------------|---------------------|
| Lines of code to add | ~100-200 | ~10-20 |
| Files to modify | 7+ | 2 |
| Risk of bugs | High | Low |
| Testing effort | Extensive | Minimal |
| Gymnasium compatibility | Breaks contract | Full compliance |
| User benefit | Can switch modes at runtime | Can use all modes at init |
| Maintenance burden | Ongoing | One-time fix |

---

## 6. Recommendation

**Implement Option B**: Fix the bug to properly respect the `render_mode` parameter.

### Rationale

1. **The current code is simply broken** - the `render_mode` parameter is advertised but ignored
2. **Gymnasium's design is intentional** - they made render_mode immutable for good reasons
3. **Users don't typically need dynamic switching** - creating separate envs for training vs eval is the standard pattern
4. **Option A adds significant complexity** for a rarely-needed feature
5. **Ecosystem alignment** - tools like Stable-Baselines3, CleanRL, etc. all expect Gymnasium semantics

### Migration Path for Users

If users currently rely on the hardcoded `"rgb_array"` behavior:

```python
# Before (implicit rgb_array):
env = plangym.make("CartPole-v1")

# After (explicit, same behavior):
env = plangym.make("CartPole-v1", render_mode="rgb_array")

# New capability (fast training):
env = plangym.make("CartPole-v1", render_mode=None)
```

---

## 7. Implementation Checklist for Option B

- [ ] Remove line 537 in `src/plangym/core.py` (`render_mode = "rgb_array"`)
- [ ] Add render_mode validation in `__init__`
- [ ] Add validation for `return_image=True` + `render_mode=None` conflict
- [ ] Update `get_image()` to raise helpful error when `render_mode=None`
- [ ] Add tests for all render modes
- [ ] Update docstring for `render_mode` parameter
- [ ] Consider adding deprecation warning if behavior change affects users
