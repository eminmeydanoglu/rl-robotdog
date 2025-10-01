"""
Training script for the robot dog using Stable-Baselines3
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import sys

# Add parent directory to path to import custom environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.robot_dog_env import RobotDogEnv


def train(
    total_timesteps=1_000_000,
    save_dir="./models",
    log_dir="./logs",
    eval_freq=10_000,
    save_freq=50_000
):
    """
    Train the robot dog using PPO algorithm.
    
    Args:
        total_timesteps: Total number of training timesteps
        save_dir: Directory to save model checkpoints
        log_dir: Directory for tensorboard logs
        eval_freq: How often to evaluate the model
        save_freq: How often to save checkpoints
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = RobotDogEnv(render_mode=None)  # No rendering during training for speed
    
    # Wrap environment
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = RobotDogEnv(render_mode=None)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="robot_dog_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        device="auto"  # Automatically use GPU if available
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Models will be saved to: {save_dir}")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Monitor training with: tensorboard --logdir {log_dir}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, "robot_dog_final.zip")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to: {final_model_path}")
    
    return model


if __name__ == "__main__":
    # Train the model
    model = train(
        total_timesteps=1_000_000,
        save_dir="./models",
        log_dir="./logs"
    )
