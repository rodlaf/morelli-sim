# Gymnasium-Style Environment API for AeroBench

## Overview

The new environment-based architecture decouples the autopilot/agent from the simulator, following the Gymnasium (formerly OpenAI Gym) API pattern. This design separates concerns and makes it easier to:

- Implement custom control policies
- Use reinforcement learning agents
- Test different autopilots on the same scenarios
- Integrate with RL frameworks

## Architecture

### Old Design (Coupled)
```
run_f16_sim(initial_state, tmax, autopilot) 
    └─> Autopilot is embedded inside simulator
        └─> Tight coupling between simulation and control
```

### New Design (Decoupled)
```
Environment (F16Env)          Agent (AutopilotAgent)
    │                              │
    ├─ Physics simulation          ├─ Decision making
    ├─ State management            ├─ Control policy
    └─ Integration                 └─ Mode tracking
         │                              │
         └──────── step(action) ────────┘
```

## Core Components

### 1. F16Env (Environment)
Manages the F-16 flight dynamics simulation.

**Key Methods:**
- `reset(initial_state=None, initial_time=0.0)` - Reset to initial conditions
- `step(u_ref)` - Advance one time step with control input
- `get_history()` - Retrieve complete simulation history

**Returns from step():**
```python
state, reward, terminated, truncated, info = env.step(u_ref)
```

### 2. AutopilotAgent (Agent Wrapper)
Wraps existing Autopilot classes to work with the new API.

**Key Methods:**
- `get_action(state, time)` - Get control from autopilot
- `is_done(state, time)` - Check if task complete
- `reset()` - Reset autopilot state

### 3. Runner Functions
- `run_f16_env(env, agent, tmax)` - Run simulation with env/agent
- `run_f16_sim_compat(...)` - Backward-compatible wrapper

## Usage Examples

### Basic Usage
```python
from aerobench.envs.f16_env import F16Env
from aerobench.envs.agents import AutopilotAgent
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot

# Create environment
env = F16Env(
    initial_state=init,
    step_size=1/30,
    extended_states=True
)

# Create agent from existing autopilot
autopilot = WaypointAutopilot(waypoints)
agent = AutopilotAgent(autopilot)

# Reset
state, info = env.reset()

# Run simulation loop
while env.time < tmax:
    u_ref = agent.get_action(state, env.time)
    state, reward, terminated, truncated, info = env.step(u_ref)
    
    if terminated or truncated:
        break
```

### Custom Agent
```python
class MyAgent:
    def get_action(self, state, time):
        # Your control logic here
        nz_cmd = ...
        ps_cmd = ...
        ny_r_cmd = ...
        throttle = ...
        return np.array([nz_cmd, ps_cmd, ny_r_cmd, throttle])
    
    def is_done(self, state, time):
        # Your termination logic
        return False

agent = MyAgent()
result = run_f16_env(env, agent, tmax=150.0)
```

### Using the Runner
```python
from aerobench.envs.runner import run_f16_env

result = run_f16_env(
    env=env,
    agent=agent,
    tmax=150.0,
    print_mode_changes=True
)

print(f"Completed {len(result['times'])} frames")
print(f"Final altitude: {result['states'][-1][12]} ft")
```

### Backward Compatibility
```python
from aerobench.envs.runner import run_f16_sim_compat

# Works exactly like old run_f16_sim
result = run_f16_sim_compat(
    initial_state=init,
    tmax=150,
    autopilot=waypoint_ap,
    step=1/30,
    extended_states=True
)
```

## Benefits

### Separation of Concerns
- **Environment**: Handles physics, integration, state management
- **Agent**: Handles decision making, control policy
- **No coupling**: Changes to one don't affect the other

### Flexibility
- Easy to swap agents without changing simulation
- Easy to test same agent on different scenarios
- Can implement custom termination/reward logic

### RL-Friendly
- Standard Gymnasium interface
- Ready for RL frameworks (Stable-Baselines3, RLlib, etc.)
- Reward shaping can be added in environment subclasses

### Testing
- Unit test agents independently
- Unit test environments independently
- Mock agents for environment testing
- Mock environments for agent testing

## Migration Guide

### Old Code
```python
from aerobench.run_f16_sim import run_f16_sim

result = run_f16_sim(
    initial_state=init,
    tmax=150,
    ap=autopilot,
    step=1/30,
    extended_states=True
)
```

### New Code (Option 1: Full Migration)
```python
from aerobench.envs.f16_env import F16Env
from aerobench.envs.agents import AutopilotAgent
from aerobench.envs.runner import run_f16_env

env = F16Env(init, step_size=1/30, extended_states=True)
agent = AutopilotAgent(autopilot)
result = run_f16_env(env, agent, tmax=150)
```

### New Code (Option 2: Compatibility Wrapper)
```python
from aerobench.envs.runner import run_f16_sim_compat

result = run_f16_sim_compat(
    initial_state=init,
    tmax=150,
    autopilot=autopilot,
    step=1/30,
    extended_states=True
)
```

## Result Format

Both old and new APIs return the same result dictionary:

```python
{
    'times': list,           # Time history
    'states': np.ndarray,    # State history (N x state_dim)
    'modes': list,           # Mode history
    'runtime': float,        # Wall-clock time
    'status': str,           # 'finished', 'terminated', etc.
    
    # If extended_states=True:
    'xd_list': list,         # State derivatives
    'u_list': list,          # Control inputs
    'Nz_list': list,         # Normal load factor
    'ps_list': list,         # Roll rate
    'Ny_r_list': list,       # Lateral load factor
}
```

## See Also

- `code/aerobench/examples/gym_api_example.py` - Complete examples
- `code/aerobench/envs/f16_env.py` - Environment implementation
- `code/aerobench/envs/agents.py` - Agent wrapper
- `code/aerobench/envs/runner.py` - Runner utilities
