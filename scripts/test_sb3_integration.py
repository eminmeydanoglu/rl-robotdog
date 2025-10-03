"""
Stable-Baselines3 Integration Test

Tests compatibility of TankEnv with Stable-Baselines3 (SB3) RL library.
Verifies that the environment can be used with standard RL algorithms
like PPO and DQN without issues.

Tests performed:
1. SB3's environment checker (validates Gymnasium interface)
2. PPO model creation
3. DQN model creation
4. Short PPO training run
5. Short DQN training run
6. Model evaluation

These tests ensure the environment is properly implemented and
compatible with the entire SB3 ecosystem.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from envs.tank_env import TankEnv


def test_env_checker():
    """Test environment with Stable-Baselines3's env checker"""
    print("\n" + "=" * 60)
    print("TEST 1: Stable-Baselines3 Environment Checker")
    print("=" * 60)
    
    print("\nRunning check_env()...")
    env = TankEnv()
    
    try:
        check_env(env, warn=True)
        print("   ✓ Environment passed all checks!")
    except Exception as e:
        print(f"   ✗ Environment check failed: {e}")
        return False
    finally:
        env.close()
    
    return True


def test_ppo_creation():
    """Test PPO model creation"""
    print("\n" + "=" * 60)
    print("TEST 2: PPO Model Creation")
    print("=" * 60)
    
    print("\nCreating PPO model...")
    env = TankEnv()
    
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64
        )
        print("   ✓ PPO model created successfully")
        print(f"   - Policy type: {type(model.policy).__name__}")
        print(f"   - Action space: {model.action_space}")
        print(f"   - Observation space: {model.observation_space}")
    except Exception as e:
        print(f"   ✗ PPO creation failed: {e}")
        return False
    finally:
        env.close()
    
    return True


def test_dqn_creation():
    """Test DQN model creation"""
    print("\n" + "=" * 60)
    print("TEST 3: DQN Model Creation")
    print("=" * 60)
    
    print("\nCreating DQN model...")
    env = TankEnv()
    
    try:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000
        )
        print("   ✓ DQN model created successfully")
        print(f"   - Policy type: {type(model.policy).__name__}")
        print(f"   - Buffer size: {model.buffer_size}")
    except Exception as e:
        print(f"   ✗ DQN creation failed: {e}")
        return False
    finally:
        env.close()
    
    return True


def test_ppo_learning():
    """Test PPO learning for a few steps"""
    print("\n" + "=" * 60)
    print("TEST 4: PPO Short Training")
    print("=" * 60)
    
    print("\nTraining PPO for 1000 steps...")
    env = TankEnv(opponent_policy='stationary')
    
    try:
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=1000)
        print("   ✓ PPO trained for 1000 steps")
        
        # Test prediction
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"   ✓ Model prediction: action={action}")
        
    except Exception as e:
        print(f"   ✗ PPO training failed: {e}")
        return False
    finally:
        env.close()
    
    return True


def test_dqn_learning():
    """Test DQN learning for a few steps"""
    print("\n" + "=" * 60)
    print("TEST 5: DQN Short Training")
    print("=" * 60)
    
    print("\nTraining DQN for 2000 steps...")
    env = TankEnv(opponent_policy='stationary')
    
    try:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            learning_starts=500,
            buffer_size=10000
        )
        model.learn(total_timesteps=2000)
        print("   ✓ DQN trained for 2000 steps")
        
        # Test prediction
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"   ✓ Model prediction: action={action}")
        
    except Exception as e:
        print(f"   ✗ DQN training failed: {e}")
        return False
    finally:
        env.close()
    
    return True


def test_model_evaluation():
    """Test model evaluation"""
    print("\n" + "=" * 60)
    print("TEST 6: Model Evaluation")
    print("=" * 60)
    
    print("\nTraining a quick model...")
    env = TankEnv(opponent_policy='stationary')
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=2000)
    
    print("\nEvaluating model for 5 episodes...")
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"   Episode {ep+1}: {steps:3d} steps, reward: {episode_reward:6.2f}")
    
    print(f"\n   ✓ Evaluation complete")
    print(f"   - Mean reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"   - Mean length: {sum(episode_lengths)/len(episode_lengths):.1f}")
    
    env.close()
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("STABLE-BASELINES3 INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Environment Checker", test_env_checker),
        ("PPO Model Creation", test_ppo_creation),
        ("DQN Model Creation", test_dqn_creation),
        ("PPO Short Training", test_ppo_learning),
        ("DQN Short Training", test_dqn_learning),
        ("Model Evaluation", test_model_evaluation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n   ✗ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
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
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n✅ TankEnv is fully compatible with Stable-Baselines3!")
        print("\nReady for:")
        print("  - Bölüm 3: Full training pipeline")
        print("  - DQN vs PPO experiments")
        print("  - Reward shaping experiments")
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed")
    
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
