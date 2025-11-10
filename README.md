# AeroBenchVVPython - F-16 Waypoint Environment

Fast C-based F-16 simulation with PufferLib integration for vectorized RL training.

## Installation

```bash
# Install system dependencies (macOS)
brew install raylib pkg-config

# Linux alternative
# sudo apt-get install libraylib-dev

# Install Python package
pip install -e .
```

This installs all dependencies (numpy, gymnasium, pufferlib) and builds the C extension.

## Quick Start

### Vectorized RL Training

```python
from aerobench.f16_waypoint import F16Waypoint

# Create vectorized environment
env = F16Waypoint(num_envs=128, step_size=1/30, time_limit=100.0, seed=42)

# Reset
obs, info = env.reset()

# Training loop
for _ in range(1000):
    actions = policy(obs)  # Your RL policy
    obs, rewards, terminals, truncations, info = env.step(actions)
    
    # Log statistics periodically
    if info:
        print(f"Episode stats: {info[0]}")

env.close()
```

### Speed Test

```bash
python -m aerobench.f16_waypoint
```

Runs 1024 parallel environments for performance benchmarking.

### Standalone C Demo

```bash
cd aerobench
gcc -o f16_waypoint f16_waypoint.c $(pkg-config --cflags --libs raylib) -lm -O3
./f16_waypoint
```

No Python required. Uses hand-coded autopilot. Press ESC to exit, drag to rotate camera, scroll to zoom.

## Environment Specification

### Action Space (4D continuous)

- [0] Nz: Normal acceleration [-2.0, 6.0] G's
- [1] ps: Roll rate [-3.0, 3.0] rad/s
- [2] Ny_r: Lateral acceleration [-1.0, 1.0] G's
- [3] throttle: Engine throttle [0.0, 1.0]

### Observation Space (28D continuous)

- [0-18] F16 state (sin/cos angles, velocities, rates, altitudes)
- [19-21] Previous actions (Nz, ps, Ny_r)
- [22-27] Waypoint in spherical coordinates (azimuth, elevation, symlog range)

See `OBSERVATION_SPACE.md` for detailed breakdown.

### Reward Function

```
reward = waypoint_proximity × velocity_regulation - penalties

where:
  waypoint_proximity = 2.0 × (range_reward + 0.1 × (azimuth + elevation))
  velocity_regulation = gaussian((velocity - 500) / 100)
  penalties = 5e-3 × (roll² + p² + q² + r²) + action_penalties
```

### Episode Termination

1. Success: Aircraft within 500ft of waypoint (terminal=1, new waypoint generated)
2. Physics violation: Altitude/airspeed out of bounds (terminal=1, full reset)
3. Timeout: Time exceeds 100 seconds (terminal=1, full reset)

## Advanced Usage

### Direct C Binding

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

# Step
binding.vec_reset(vec_env, seed=42)
actions[:] = [[1.0, 0.0, 0.0, 0.5]] * num_envs
binding.vec_step(vec_env)

# Get statistics
logs = binding.vec_log(vec_env)  # Returns dict with perf, score, episode_return, etc.

# Cleanup
binding.vec_close(vec_env)
```

## Architecture

```
f16_waypoint.py (PufferEnv wrapper)
    ↓
binding.so (C/Python interface via env_binding.h)
    ↓
f16_waypoint.h (Environment logic + reward + observations)
    ├→ f16_model.h (F16 physics simulation)
    └→ raylib_renderer.h (3D visualization)
```

## Testing

```bash
python test_binding.py
```

Verifies both single and vectorized environment functionality.

## Files

- `aerobench/f16_waypoint.py` - PufferLib environment wrapper
- `aerobench/binding.c` - C binding implementation
- `aerobench/f16_waypoint.h` - Core environment (reset, step, reward, observations)
- `aerobench/f16_model.h` - F16 physics model
- `aerobench/raylib_renderer.h` - 3D renderer
- `env_binding.h` - PufferLib generic binding header
- `test_binding.py` - Integration tests

## Citation

"Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, ARCH 2018

Distribution A: Approved for Public Release (88ABW-2020-2188)

