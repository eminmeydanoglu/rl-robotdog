"""
Evaluation script to test trained models
"""

import gymnasium as gym
from stable_baselines3 import PPO
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.robot_dog_env import RobotDogEnv


def evaluate(
    model_path,
    num_episodes=10,
    render=True,
    deterministic=True
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
    """
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = RobotDogEnv(render_mode=render_mode)
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(1./240.)  # Match simulation timestep
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Average Length: {sum(episode_lengths)/len(episode_lengths):.2f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained robot dog model")
    parser.add_argument("--model", type=str, default="./models/robot_dog_final.zip",
                        help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions instead of deterministic")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic
    )
