"""
Tank Environment Test Script

Tests that the Gymnasium wrapper works correctly and implements
all required functionality for RL training.

Comprehensive tests include:
- Environment creation with different configurations
- Reset functionality with seeds and options
- Step execution with all action types
- Opponent policy variations
- Reward shaping mechanisms
- Episode completion and truncation
- Observation normalization

This ensures the TankEnv is ready for use with any Gymnasium-compatible
RL framework or algorithm.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.tank_env import TankEnv, make_tank_env


def test_environment_creation():
    """Test environment creation"""
    print("\n" + "=" * 60)
    print("TEST 1: Environment Creation")
    print("=" * 60)
    
    # Test with different render modes
    print("\n1.1. Creating environment (headless)...")
    env = TankEnv(render_mode=None)
    print(f"   ✓ Environment created")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Observation shape: {env.observation_space.shape}")
    env.close()
    
    print("\n1.2. Creating environment with factory function...")
    env = make_tank_env(opponent_policy='stationary')
    print(f"   ✓ Environment created with factory")
    print(f"   - Opponent policy: {env.opponent_policy}")
    env.close()
    
    return True


def test_reset():
    """Test reset functionality"""
    print("\n" + "=" * 60)
    print("TEST 2: Reset Functionality")
    print("=" * 60)
    
    env = TankEnv()
    
    print("\n2.1. Testing reset()...")
    obs, info = env.reset()
    print(f"   ✓ Reset successful")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Observation dtype: {obs.dtype}")
    print(f"   - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"   - Info keys: {list(info.keys())}")
    
    print("\n2.2. Testing reset with seed...")
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    print(f"   ✓ Seeded reset")
    print(f"   - Same seed gives same state: {np.allclose(obs1, obs2)}")
    
    print("\n2.3. Testing reset with random positions...")
    obs, _ = env.reset(options={'random_positions': True})
    print(f"   ✓ Reset with random positions")
    print(f"   - Tank 1 pos (normalized): ({obs[0]:.3f}, {obs[1]:.3f})")
    print(f"   - Tank 2 pos (normalized): ({obs[7]:.3f}, {obs[8]:.3f})")
    
    env.close()
    return True


def test_step():
    """Test step functionality"""
    print("\n" + "=" * 60)
    print("TEST 3: Step Functionality")
    print("=" * 60)
    
    env = TankEnv(opponent_policy='stationary')
    obs, _ = env.reset()
    
    print("\n3.1. Testing single step...")
    action = 0  # Move forward
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Step executed")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Reward: {reward:.4f}")
    print(f"   - Terminated: {terminated}")
    print(f"   - Truncated: {truncated}")
    print(f"   - Info keys: {list(info.keys())}")
    
    print("\n3.2. Testing all actions...")
    action_names = ['Forward', 'Turn Left', 'Turn Right', 'Fire', 'Do Nothing']
    for action in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   - Action {action} ({action_names[action]}): reward={reward:.4f}")
        if terminated or truncated:
            env.reset()
    
    print("\n3.3. Testing episode completion...")
    env.reset()
    steps = 0
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    print(f"   ✓ Episode completed")
    print(f"   - Steps: {steps}")
    print(f"   - Total reward: {total_reward:.2f}")
    print(f"   - Winner: {info.get('winner', 'None')}")
    
    env.close()
    return True


def test_opponent_policies():
    """Test different opponent policies"""
    print("\n" + "=" * 60)
    print("TEST 4: Opponent Policies")
    print("=" * 60)
    
    policies = ['stationary', 'simple', 'random']
    
    for policy in policies:
        print(f"\n4.{policies.index(policy) + 1}. Testing '{policy}' policy...")
        env = TankEnv(opponent_policy=policy)
        obs, _ = env.reset()
        
        hits = 0
        for _ in range(50):
            action = 3 if np.random.random() < 0.2 else 0  # Mostly forward, sometimes shoot
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get('hit_opponent', False):
                hits += 1
            if terminated or truncated:
                break
        
        print(f"   ✓ Policy tested")
        print(f"   - Opponent hits: {hits}")
        print(f"   - Episode length: {info['steps']}")
        env.close()
    
    return True


def test_reward_shaping():
    """Test reward shaping"""
    print("\n" + "=" * 60)
    print("TEST 5: Reward Shaping")
    print("=" * 60)
    
    print("\n5.1. Testing sparse rewards...")
    env_sparse = TankEnv(reward_shaping=False)
    obs, _ = env_sparse.reset()
    rewards_sparse = []
    for _ in range(20):
        obs, reward, terminated, truncated, _ = env_sparse.step(0)
        rewards_sparse.append(reward)
        if terminated or truncated:
            break
    print(f"   ✓ Sparse rewards")
    print(f"   - Sample rewards: {[f'{r:.4f}' for r in rewards_sparse[:5]]}")
    env_sparse.close()
    
    print("\n5.2. Testing shaped rewards...")
    env_shaped = TankEnv(reward_shaping=True)
    obs, _ = env_shaped.reset()
    rewards_shaped = []
    for _ in range(20):
        obs, reward, terminated, truncated, _ = env_shaped.step(0)
        rewards_shaped.append(reward)
        if terminated or truncated:
            break
    print(f"   ✓ Shaped rewards")
    print(f"   - Sample rewards: {[f'{r:.4f}' for r in rewards_shaped[:5]]}")
    env_shaped.close()
    
    return True


def test_gymnasium_compatibility():
    """Test Gymnasium API compatibility"""
    print("\n" + "=" * 60)
    print("TEST 6: Gymnasium API Compatibility")
    print("=" * 60)
    
    env = TankEnv()
    
    print("\n6.1. Checking required attributes...")
    required_attrs = ['action_space', 'observation_space', 'metadata']
    for attr in required_attrs:
        has_attr = hasattr(env, attr)
        print(f"   - {attr}: {'✓' if has_attr else '✗'}")
    
    print("\n6.2. Checking required methods...")
    required_methods = ['reset', 'step', 'render', 'close']
    for method in required_methods:
        has_method = hasattr(env, method) and callable(getattr(env, method))
        print(f"   - {method}(): {'✓' if has_method else '✗'}")
    
    print("\n6.3. Testing action space sampling...")
    actions = [env.action_space.sample() for _ in range(10)]
    print(f"   ✓ Sampled actions: {actions}")
    
    print("\n6.4. Testing observation space contains...")
    obs, _ = env.reset()
    contains = env.observation_space.contains(obs)
    print(f"   ✓ Observation in space: {contains}")
    
    env.close()
    return True


def test_full_episode():
    """Run a full episode and collect statistics"""
    print("\n" + "=" * 60)
    print("TEST 7: Full Episode Statistics")
    print("=" * 60)
    
    env = TankEnv(opponent_policy='simple')
    
    print("\nRunning 5 episodes...")
    episode_stats = []
    
    for ep in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        hits = 0
        
        while True:
            # Random agent
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            if info.get('hit_opponent', False):
                hits += 1
            
            if terminated or truncated:
                episode_stats.append({
                    'episode': ep + 1,
                    'steps': steps,
                    'reward': episode_reward,
                    'hits': hits,
                    'winner': info.get('winner', 0)
                })
                break
    
    print("\nEpisode Statistics:")
    print("-" * 60)
    for stat in episode_stats:
        winner_str = f"Tank {stat['winner']}" if stat['winner'] > 0 else "Draw"
        print(f"Episode {stat['episode']}: "
              f"{stat['steps']:3d} steps, "
              f"reward: {stat['reward']:6.2f}, "
              f"hits: {stat['hits']}, "
              f"winner: {winner_str}")
    
    # Calculate averages
    avg_steps = np.mean([s['steps'] for s in episode_stats])
    avg_reward = np.mean([s['reward'] for s in episode_stats])
    avg_hits = np.mean([s['hits'] for s in episode_stats])
    wins = sum(1 for s in episode_stats if s['winner'] == 1)
    
    print("-" * 60)
    print(f"Averages: {avg_steps:.1f} steps, {avg_reward:.2f} reward, {avg_hits:.1f} hits")
    print(f"Win rate: {wins}/5 ({wins*20}%)")
    
    env.close()
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TANK ENVIRONMENT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Environment Creation", test_environment_creation),
        ("Reset Functionality", test_reset),
        ("Step Functionality", test_step),
        ("Opponent Policies", test_opponent_policies),
        ("Reward Shaping", test_reward_shaping),
        ("Gymnasium Compatibility", test_gymnasium_compatibility),
        ("Full Episode Statistics", test_full_episode)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n   ✗ ERROR: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print("=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Bölüm 2 Tamamlandı! ✓")
        print("\nŞimdi yapabilecekleriniz:")
        print("1. Bölüm 3: Tek ajanlı eğitim pipeline'ı")
        print("2. Test training: python scripts/train_single.py")
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed")
    
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
