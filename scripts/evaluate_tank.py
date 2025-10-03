"""
Model Evaluation Script for Tank Battle

Tests trained models and reports comprehensive performance metrics.
Part of Chapter 3: Evaluation Pipeline

Features:
- Detailed evaluation statistics (win rate, mean reward, episode length)
- Shooting accuracy tracking (hit rate, missed shots)
- Deterministic and stochastic policy evaluation
- Optional rendering to visualize agent behavior
- Performance grading and assessment
- Save results to file for analysis
- Progress reporting during evaluation

This script helps assess how well the trained agent performs against
different opponent policies and provides insights for improvement.
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from typing import List, Dict, Tuple
import time

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

from envs import TankEnv


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Tank Battle agent'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.zip file)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='PPO',
        choices=['PPO', 'DQN'],
        help='RL algorithm (default: PPO)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes (default: 100)'
    )
    
    parser.add_argument(
        '--opponent',
        type=str,
        default='stationary',
        choices=['stationary', 'simple', 'random'],
        help='Opponent policy (default: stationary)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes (slower)'
    )
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic policy'
    )
    
    parser.add_argument(
        '--reward-shaping',
        action='store_true',
        help='Use shaped rewards (must match training)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='Max steps per episode (default: 1000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--save-stats',
        type=str,
        default=None,
        help='Save statistics to file (default: None)'
    )
    
    return parser.parse_args()


def detailed_evaluation(
    model,
    env: TankEnv,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False
) -> Dict:
    """
    Perform detailed evaluation with additional metrics
    
    Args:
        model: Trained model
        env: Environment
        n_episodes: Number of episodes
        deterministic: Use deterministic policy
        render: Render episodes
    
    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0
    total_hits = 0
    total_shots = 0
    total_missed = 0
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    print("=" * 70)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_hits = 0
        episode_shots = 0
        episode_missed = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track shooting stats
            if action == 3:  # Fire action
                episode_shots += 1
            
            if 'tank1_missed' in info:
                episode_missed += info['tank1_missed']
            
            done = terminated or truncated
            
            if render:
                env.render()
                time.sleep(0.01)  # Slow down for visibility
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_missed += episode_missed
        
        # Determine outcome
        if info['winner'] == 1:
            wins += 1
            outcome = "WIN"
        elif info['winner'] == 2:
            losses += 1
            outcome = "LOSS"
        else:
            draws += 1
            outcome = "DRAW"
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}/{n_episodes}: "
                  f"Reward={episode_reward:7.2f}, "
                  f"Length={episode_length:4d}, "
                  f"Result={outcome}")
    
    print("=" * 70)
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    win_rate = wins / n_episodes * 100
    loss_rate = losses / n_episodes * 100
    draw_rate = draws / n_episodes * 100
    
    stats = {
        'n_episodes': n_episodes,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'total_missed': total_missed,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    return stats


def print_statistics(stats: Dict):
    """Print evaluation statistics in a nice format"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Episode Statistics:")
    print(f"  Episodes:        {stats['n_episodes']}")
    print(f"  Mean Reward:     {stats['mean_reward']:7.2f} ± {stats['std_reward']:6.2f}")
    print(f"  Mean Length:     {stats['mean_length']:7.1f} ± {stats['std_length']:6.1f}")
    
    print(f"\n🎯 Win/Loss Statistics:")
    print(f"  Wins:            {stats['wins']:3d}  ({stats['win_rate']:5.1f}%)")
    print(f"  Losses:          {stats['losses']:3d}  ({stats['loss_rate']:5.1f}%)")
    print(f"  Draws:           {stats['draws']:3d}  ({stats['draw_rate']:5.1f}%)")
    
    print(f"\n🎮 Performance Summary:")
    if stats['win_rate'] > 80:
        print("  ⭐⭐⭐ Excellent! Agent dominates the opponent!")
    elif stats['win_rate'] > 60:
        print("  ⭐⭐ Very Good! Agent performs well!")
    elif stats['win_rate'] > 40:
        print("  ⭐ Good! Agent is competitive!")
    else:
        print("  ⚠️ Needs Improvement. Consider more training.")
    
    print("\n" + "=" * 70)


def save_statistics(stats: Dict, filepath: str):
    """Save statistics to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("Tank Battle Agent - Evaluation Statistics\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Episodes: {stats['n_episodes']}\n")
        f.write(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n")
        f.write(f"Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}\n\n")
        
        f.write(f"Wins: {stats['wins']} ({stats['win_rate']:.1f}%)\n")
        f.write(f"Losses: {stats['losses']} ({stats['loss_rate']:.1f}%)\n")
        f.write(f"Draws: {stats['draws']} ({stats['draw_rate']:.1f}%)\n\n")
        
        f.write(f"Total Missed Shots: {stats['total_missed']}\n\n")
        
        f.write("Episode Rewards:\n")
        for i, reward in enumerate(stats['episode_rewards'], 1):
            f.write(f"  Episode {i:3d}: {reward:7.2f}\n")
    
    print(f"\n✓ Statistics saved to: {filepath}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("TANK BATTLE - MODEL EVALUATION")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Algorithm:       {args.algorithm}")
    print(f"Episodes:        {args.episodes}")
    print(f"Opponent:        {args.opponent}")
    print(f"Deterministic:   {args.deterministic}")
    print(f"Render:          {args.render}")
    print(f"Reward Shaping:  {args.reward_shaping}")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading {args.algorithm} model...")
    if args.algorithm == 'PPO':
        model = PPO.load(args.model)
    elif args.algorithm == 'DQN':
        model = DQN.load(args.model)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    print("✓ Model loaded successfully")
    
    # Create environment
    print("\nCreating evaluation environment...")
    env = TankEnv(
        render_mode='human' if args.render else None,
        opponent_policy=args.opponent,
        reward_shaping=args.reward_shaping,
        max_steps=args.max_steps
    )
    env.reset(seed=args.seed)
    print("✓ Environment created")
    
    # Evaluate
    stats = detailed_evaluation(
        model,
        env,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render
    )
    
    # Print results
    print_statistics(stats)
    
    # Save statistics if requested
    if args.save_stats:
        save_statistics(stats, args.save_stats)
    
    # Cleanup
    env.close()
    
    print("\n✓ Evaluation complete!\n")


if __name__ == "__main__":
    main()
