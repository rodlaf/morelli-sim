#!/usr/bin/env python3
"""Test script to verify PufferLib binding works correctly"""

import numpy as np
import sys

def test_basic_binding():
    """Test basic binding functionality"""
    print("=" * 60)
    print("Testing aerobench.binding module")
    print("=" * 60)
    
    try:
        from aerobench import binding
        print("✓ Module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import binding: {e}")
        print("\nYou need to build the extension first:")
        print("  python setup.py build_ext --inplace")
        return False
    
    # Test single environment
    print("\nTesting single environment...")
    obs = np.zeros(28, dtype=np.float32)
    actions = np.zeros(4, dtype=np.float32)
    rewards = np.zeros(1, dtype=np.float32)
    terminals = np.zeros(1, dtype=np.uint8)
    truncations = np.zeros(1, dtype=np.uint8)
    
    try:
        # env_init signature: (obs, actions, rewards, terminals, truncations, seed, **kwargs)
        env = binding.env_init(obs, actions, rewards, terminals, truncations, 42,
                              step_size=1/30, time_limit=100.0)
        print(f"✓ env_init succeeded, handle: {env}")
        
        binding.env_reset(env, 42)
        print(f"✓ env_reset succeeded")
        print(f"  Initial obs[0:5]: {obs[:5]}")
        
        actions[:] = [1.0, 0.0, 0.0, 0.5]  # Test action
        binding.env_step(env)
        print(f"✓ env_step succeeded")
        print(f"  Reward: {rewards[0]:.4f}")
        print(f"  Terminal: {terminals[0]}")
        
        binding.env_close(env)
        print(f"✓ env_close succeeded")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test vectorized environments
    print("\nTesting vectorized environments...")
    num_envs = 4
    obs_vec = np.zeros((num_envs, 28), dtype=np.float32)
    actions_vec = np.zeros((num_envs, 4), dtype=np.float32)
    rewards_vec = np.zeros(num_envs, dtype=np.float32)
    terminals_vec = np.zeros(num_envs, dtype=np.uint8)
    truncations_vec = np.zeros(num_envs, dtype=np.uint8)
    
    try:
        # vec_init signature: (obs, actions, rewards, terminals, truncations, num_envs, seed, **kwargs)
        vec_env = binding.vec_init(obs_vec, actions_vec, rewards_vec, 
                                   terminals_vec, truncations_vec, num_envs, 42,
                                   step_size=1/30, time_limit=100.0)
        print(f"✓ vec_init succeeded, handle: {vec_env}")
        
        binding.vec_reset(vec_env, 42)
        print(f"✓ vec_reset succeeded")
        print(f"  Env 0 obs[0:5]: {obs_vec[0, :5]}")
        
        actions_vec[:] = [[1.0, 0.0, 0.0, 0.5]] * num_envs
        binding.vec_step(vec_env)
        print(f"✓ vec_step succeeded")
        print(f"  Rewards: {rewards_vec}")
        
        logs = binding.vec_log(vec_env)
        print(f"✓ vec_log succeeded")
        print(f"  Log keys: {list(logs.keys())}")
        
        binding.vec_close(vec_env)
        print(f"✓ vec_close succeeded")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


def test_pufferlib_env():
    """Test PufferLib environment wrapper"""
    print("\n" + "=" * 60)
    print("Testing F16Waypoint PufferEnv")
    print("=" * 60)
    
    try:
        import pufferlib
        print("✓ PufferLib imported")
    except ImportError:
        print("✗ PufferLib not installed")
        print("\nInstall with: pip install pufferlib")
        return False
    
    try:
        from aerobench.f16_waypoint import F16Waypoint
        print("✓ F16Waypoint imported")
        
        env = F16Waypoint(num_envs=4, seed=42)
        print(f"✓ Environment created")
        print(f"  Observation space: {env.single_observation_space}")
        print(f"  Action space: {env.single_action_space}")
        
        obs, info = env.reset()
        print(f"✓ Reset succeeded")
        print(f"  Observation shape: {obs.shape}")
        
        actions = np.random.uniform(
            low=env.single_action_space.low,
            high=env.single_action_space.high,
            size=(4, 4)
        ).astype(np.float32)
        
        obs, rewards, terminals, truncations, info = env.step(actions)
        print(f"✓ Step succeeded")
        print(f"  Rewards: {rewards}")
        
        env.close()
        print(f"✓ Close succeeded")
        
        print("\n" + "=" * 60)
        print("PufferEnv test passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_basic_binding()
    
    if success:
        test_pufferlib_env()
    
    sys.exit(0 if success else 1)
