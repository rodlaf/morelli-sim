# F16 Waypoint PufferLib Integration

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy gymnasium pufferlib
```

### 2. Build the Extension

```bash
python setup.py build_ext --inplace
```

This will compile:
- `aerobench/binding.cpython-*.so` - PufferLib C binding
- `aerobench/f16_waypoint_cy.cpython-*.so` - Cython wrapper (legacy)

### 3. Test the Installation

```bash
python test_binding.py
```

This will verify:
- Basic binding functions work (env_init, env_reset, env_step, env_close)
- Vectorized functions work (vec_init, vec_reset, vec_step, vec_log, vec_close)
- PufferEnv wrapper works correctly

### 4. Run Speed Test

```bash
python -m aerobench.f16_waypoint
```

This will run a vectorized speed test with 1024 parallel environments.

## Usage

### Basic Usage (PufferLib)

```python
from aerobench.f16_waypoint import F16Waypoint

# Create vectorized environment
env = F16Waypoint(num_envs=128, step_size=1/30, time_limit=100.0, seed=42)

# Reset
obs, info = env.reset()

# Step
for _ in range(1000):
    actions = policy(obs)  # Your policy here
    obs, rewards, terminals, truncations, info = env.step(actions)
    
    # Check for episode logs
    if info:
        print(f"Episode stats: {info[0]}")

env.close()
```

### Direct Binding Usage (Advanced)

```python
from aerobench import binding
import numpy as np

# Allocate buffers
num_envs = 4
obs = np.zeros((num_envs, 28), dtype=np.float32)
actions = np.zeros((num_envs, 4), dtype=np.float32)
rewards = np.zeros(num_envs, dtype=np.float32)
terminals = np.zeros(num_envs, dtype=np.uint8)
truncations = np.zeros(num_envs, dtype=np.uint8)

# Initialize
vec_env = binding.vec_init(obs, actions, rewards, terminals, truncations,
                           num_envs, seed=42, step_size=1/30, time_limit=100.0)

# Reset
binding.vec_reset(vec_env, seed=42)

# Step
actions[:] = [[1.0, 0.0, 0.0, 0.5]] * num_envs
binding.vec_step(vec_env)

# Get logs
logs = binding.vec_log(vec_env)
print(logs)  # {'perf': ..., 'score': ..., 'episode_return': ..., 'episode_length': ..., 'n': ...}

# Clean up
binding.vec_close(vec_env)
```

## Spaces

### Observation Space (28D continuous)

See `OBSERVATION_SPACE.md` for detailed description.

- **[0-18]** F16 state (sin/cos angles, velocities, rates, altitudes)
- **[19-21]** Previous actions (Nz, ps, Ny_r)
- **[22-27]** Waypoint in spherical coordinates (azimuth, elevation, range)

### Action Space (4D continuous)

- **[0]** Nz: Normal acceleration command [-2.0, 6.0] G's
- **[1]** ps: Roll rate command [-3.0, 3.0] rad/s
- **[2]** Ny_r: Lateral acceleration command [-1.0, 1.0] G's
- **[3]** throttle: Engine throttle [0.0, 1.0]

## Architecture

```
f16_waypoint.py (PufferEnv wrapper)
    ↓
binding.so (Python/C interface via env_binding.h)
    ↓
f16_waypoint.h (Environment logic)
    ├→ f16_model.h (F16 physics simulation)
    └→ raylib_renderer.h (3D visualization)
```

## Files

- `aerobench/f16_waypoint.py` - PufferLib environment wrapper
- `aerobench/binding.c` - C binding implementation
- `env_binding.h` - PufferLib binding header (generic)
- `aerobench/f16_waypoint.h` - Core environment implementation
- `aerobench/f16_model.h` - F16 physics model
- `aerobench/raylib_renderer.h` - Raylib 3D renderer
- `test_binding.py` - Integration tests

## Troubleshooting

### Build Errors

If you get raylib include errors:
```bash
# macOS with Homebrew
brew install raylib

# Linux
sudo apt-get install libraylib-dev
```

### Import Errors

Make sure to build the extension first:
```bash
python setup.py build_ext --inplace
```

The `.so` files should appear in the `aerobench/` directory.

### Runtime Errors

Check that all dependencies are installed:
```bash
pip list | grep -E "(numpy|gymnasium|pufferlib)"
```
