"""
Single Agent Training Script

Bu script tek bir RL ajanını sabit bir rakibe karşı eğitir.
Bölüm 3: Tek Ajanlı Eğitim Pipeline
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor

from envs import TankEnv


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train a single RL agent for Tank Battle'
    )
    
    # Algorithm
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        default='PPO',
        choices=['PPO', 'DQN'],
        help='RL algorithm to use (default: PPO)'
    )
    
    # Training
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=50000,
        help='Total timesteps for training (default: 50000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    # Environment
    parser.add_argument(
        '--opponent',
        type=str,
        default='stationary',
        choices=['stationary', 'simple', 'random'],
        help='Opponent policy (default: stationary)'
    )
    
    parser.add_argument(
        '--reward-shaping',
        action='store_true',
        help='Use shaped rewards instead of sparse rewards'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='Maximum steps per episode (default: 1000)'
    )
    
    # Saving
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save models (default: models)'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10000,
        help='Save checkpoint every N steps (default: 10000)'
    )
    
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Custom run name (default: auto-generated)'
    )
    
    # Evaluation
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=5000,
        help='Evaluate every N steps (default: 5000)'
    )
    
    parser.add_argument(
        '--n-eval-episodes',
        type=int,
        default=10,
        help='Number of episodes for evaluation (default: 10)'
    )
    
    # Hyperparameters (PPO)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    
    # Misc
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level (default: 1)'
    )
    
    return parser.parse_args()


def create_env(opponent_policy: str, reward_shaping: bool, max_steps: int, seed: int = None):
    """
    Create and wrap the training environment
    
    Args:
        opponent_policy: Opponent strategy
        reward_shaping: Use shaped rewards
        max_steps: Max steps per episode
        seed: Random seed
    
    Returns:
        Wrapped environment
    """
    env = TankEnv(
        render_mode=None,  # Headless for training
        opponent_policy=opponent_policy,
        reward_shaping=reward_shaping,
        max_steps=max_steps
    )
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def create_model(algorithm: str, env, learning_rate: float, batch_size: int, 
                 seed: int = None, verbose: int = 1):
    """
    Create RL model
    
    Args:
        algorithm: 'PPO' or 'DQN'
        env: Training environment
        learning_rate: Learning rate
        batch_size: Batch size
        seed: Random seed
        verbose: Verbosity level
    
    Returns:
        Model instance
    """
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=verbose,
            seed=seed,
            tensorboard_log="./logs/"
        )
    elif algorithm == 'DQN':
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=batch_size,
            gamma=0.99,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=verbose,
            seed=seed,
            tensorboard_log="./logs/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model


def setup_callbacks(eval_env, save_dir: Path, save_freq: int, eval_freq: int, 
                    n_eval_episodes: int):
    """
    Setup training callbacks
    
    Args:
        eval_env: Evaluation environment
        save_dir: Directory to save models
        save_freq: Checkpoint frequency
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
    
    Returns:
        CallbackList
    """
    # Create directories
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    return CallbackList([checkpoint_callback, eval_callback])


def print_training_info(args, run_name: str, save_dir: Path):
    """Print training configuration"""
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Run Name:          {run_name}")
    print(f"Algorithm:         {args.algorithm}")
    print(f"Total Timesteps:   {args.timesteps:,}")
    print(f"Opponent:          {args.opponent}")
    print(f"Reward Shaping:    {args.reward_shaping}")
    print(f"Learning Rate:     {args.learning_rate}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Seed:              {args.seed if args.seed else 'Random'}")
    print(f"Save Directory:    {save_dir}")
    print(f"Save Frequency:    Every {args.save_freq:,} steps")
    print(f"Eval Frequency:    Every {args.eval_freq:,} steps")
    print("=" * 70)
    print("\n🚀 Starting training...\n")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reward_type = "shaped" if args.reward_shaping else "sparse"
        run_name = f"{args.algorithm}_{args.opponent}_{reward_type}_{timestamp}"
    else:
        run_name = args.run_name
    
    # Setup save directory
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print_training_info(args, run_name, save_dir)
    
    # Create environments
    print("Creating environments...")
    train_env = create_env(
        args.opponent, 
        args.reward_shaping, 
        args.max_steps,
        args.seed
    )
    
    eval_env = create_env(
        args.opponent,
        args.reward_shaping,
        args.max_steps,
        args.seed + 100 if args.seed else None  # Different seed for eval
    )
    print("✓ Environments created\n")
    
    # Create model
    print(f"Creating {args.algorithm} model...")
    model = create_model(
        args.algorithm,
        train_env,
        args.learning_rate,
        args.batch_size,
        args.seed,
        args.verbose
    )
    print("✓ Model created\n")
    
    # Setup callbacks
    print("Setting up callbacks...")
    callbacks = setup_callbacks(
        eval_env,
        save_dir,
        args.save_freq,
        args.eval_freq,
        args.n_eval_episodes
    )
    print("✓ Callbacks ready\n")
    
    # Save configuration
    config_file = save_dir / "config.txt"
    with open(config_file, 'w') as f:
        f.write("Training Configuration\n")
        f.write("=" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"✓ Configuration saved to {config_file}\n")
    
    # Start training
    print("=" * 70)
    print("TRAINING IN PROGRESS")
    print("=" * 70)
    print("\nMonitor with TensorBoard:")
    print(f"  tensorboard --logdir logs/")
    print(f"  Then open: http://localhost:6006\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=True
        )
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("⚠ Training interrupted by user")
        print("=" * 70)
    
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ Training failed with error: {e}")
        print("=" * 70)
        raise
    
    finally:
        # Save final model
        final_model_path = save_dir / "final_model.zip"
        model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
        # Close environments
        train_env.close()
        eval_env.close()
        
        print("\n" + "=" * 70)
        print("Training session ended")
        print("=" * 70)
        print(f"\nResults saved in: {save_dir}")
        print(f"Best model: {save_dir / 'best_model.zip'}")
        print(f"Final model: {final_model_path}")
        print(f"Checkpoints: {save_dir / 'checkpoints'}")
        print(f"\nTo evaluate:")
        print(f"  python scripts/evaluate.py --model {save_dir / 'best_model.zip'}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
