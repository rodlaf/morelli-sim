# F16 Waypoint Observation Space

This document describes the 28-dimensional observation space that exactly matches the JAX implementation for RL training.

## Overview

The observation vector is constructed by `get_observation()` in `f16_waypoint.h` and consists of three main components:

1. **F16 State** (19 values) - Aircraft dynamics
2. **Previous Actions** (3 values) - Temporal information
3. **Waypoint** (6 values) - Goal-relative information

**Total: 28 dimensions**

## Detailed Breakdown

### F16 State Observations (indices 0-18)

| Index | Description | Transformation | Range |
|-------|-------------|----------------|-------|
| 0 | Velocity (ft/s) | `VT / 1000.0` | ~0.3 - 0.7 |
| 1-2 | Angle of attack | `sin(alpha)`, `cos(alpha)` | [-1, 1] |
| 3-4 | Sideslip angle | `sin(beta)`, `cos(beta)` | [-1, 1] |
| 5-6 | Roll angle | `sin(phi)`, `cos(phi)` | [-1, 1] |
| 7-8 | Pitch angle | `sin(theta)`, `cos(theta)` | [-1, 1] |
| 9-10 | Yaw angle | `sin(psi)`, `cos(psi)` | [-1, 1] |
| 11 | Roll rate (P) | No scaling | rad/s |
| 12 | Pitch rate (Q) | No scaling | rad/s |
| 13 | Yaw rate (R) | No scaling | rad/s |
| 14 | Altitude | `ALT / 1000.0` | ~1.0 - 5.0 |
| 15 | Engine power | `POW / 10.0` | 0.5 - 1.0 |
| 16 | LQR integrator (Nz) | `state[13]` or 0 | varies |
| 17 | LQR integrator (ps) | `state[14]` or 0 | varies |
| 18 | LQR integrator (Ny_r) | `state[15]` or 0 | varies |

**Key Features:**
- **Sin/Cos encoding** eliminates angle wrapping discontinuities at ±π
- **Normalized scales** keep values roughly in [-1, 1] or [0, 1] range
- **No absolute position** (POSE, POSN) - observations are ego-centric

### Previous Action Observations (indices 19-21)

| Index | Description | Range |
|-------|-------------|-------|
| 19 | Previous Nz command | -1.0 to 4.0 |
| 20 | Previous ps command | ~-1.2 to 1.2 |
| 21 | Previous throttle | 0.0 to 1.0 |

**Purpose:** Enables temporal credit assignment and smooth action trajectories

### Waypoint Observations (indices 22-27)

| Index | Description | Transformation | Range |
|-------|-------------|----------------|-------|
| 22-23 | Azimuth error | `sin(delta_az)`, `cos(delta_az)` | [-1, 1] |
| 24-25 | Elevation error | `sin(delta_elev)`, `cos(delta_elev)` | [-1, 1] |
| 26 | Range to waypoint | `symlog(range)` | ~3.0 - 4.2 |
| 27 | Time remaining | Normalized [0, 1] | [0, 1] |

**Coordinate System:**
- **Azimuth error**: Heading error to waypoint (in aircraft's yaw frame)
- **Elevation error**: Pitch error to waypoint (in aircraft's pitch frame)
- **Range**: Distance to waypoint, compressed via symlog transformation

**Symlog transformation:** `sign(x) * log10(|x| + 1)`
- Compresses large ranges: 1000 ft → 3.0, 10000 ft → 4.0, 100000 ft → 5.0
- Preserves sign and zero

## Design Rationale

### Why Sin/Cos Encoding?

Angles like φ = -π and φ = +π are the same physical state, but as raw values they appear maximally different. Neural networks struggle with this discontinuity.

Using `[sin(φ), cos(φ)]` creates a continuous 2D circle representation where nearby angles have nearby representations.

### Why Ego-Centric (No Absolute Position)?

The task is waypoint navigation, not position tracking. Absolute coordinates:
- Add irrelevant information (agent doesn't care about grid location)
- Break translation invariance (same relative configuration looks different)
- Increase observation dimensionality unnecessarily

Relative spherical coordinates capture "where is the goal from my perspective" which is all the agent needs.

### Why Symlog for Range?

Ranges vary over orders of magnitude (100 ft to 100,000 ft). Without compression:
- Neural networks need huge dynamic range
- Gradients become unstable
- Small nearby distances get lost in noise

Symlog compresses large values while preserving small ones and zero-crossing behavior.

## Usage

### C/C++ Code
```c
F16Waypoint env;
double obs[OBS_DIM_TOTAL];  // 28 dimensions
get_observation(&env, obs);
// obs now contains RL-ready observation vector
```

### Python (via Cython)
```python
from aerobench.f16_waypoint import F16Waypoint

env = F16Waypoint()
state, info = env.reset()
obs = env.get_observation()  # Returns numpy array of shape (28,)
```

## Comparison to Raw State

| Raw State | Observation |
|-----------|-------------|
| 13-16 values | 28 values |
| Raw angles (discontinuous) | Sin/cos angles (continuous) |
| Absolute position | Relative waypoint info |
| No history | Previous actions included |
| Unbounded ranges | Scaled/compressed values |
| For physics simulation | For RL training |

The autopilot still uses raw state internally. The observation function is only needed for training RL policies.
