"""
Tank Environment - Gymnasium Wrapper

Bu modül, TankGameEngine'i Gymnasium interface'i ile sarmalayarak
Stable-Baselines3 gibi RL kütüphaneleri ile uyumlu hale getirir.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from envs.tank_game_engine import TankGameEngine


class TankEnv(gym.Env):
    """
    Gymnasium Environment for Tank Battle
    
    Single-agent environment where one tank is controlled by the RL agent
    and the other tank is controlled by a fixed opponent (or another agent).
    
    Action Space:
        Discrete(5):
        - 0: Move forward
        - 1: Turn left
        - 2: Turn right
        - 3: Fire
        - 4: Do nothing
    
    Observation Space:
        Box(26,):
        - Tank 1 state (7): x, y, angle, health, alive, bullet_count, fire_cooldown
        - Tank 2 state (7): x, y, angle, health, alive, bullet_count, fire_cooldown
        - Tank 1 bullets (6): x1, y1, x2, y2, x3, y3 (normalized)
        - Tank 2 bullets (6): x1, y1, x2, y2, x3, y3 (normalized)
    
    Reward:
        - Hit opponent: +10
        - Win (defeat opponent): +100
        - Lose (defeated by opponent): -100
        - Each step: -0.01 (small penalty to encourage faster wins)
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        opponent_policy: str = 'random',
        max_steps: int = 1000,
        reward_shaping: bool = False
    ):
        """
        Initialize Tank Environment
        
        Args:
            render_mode: 'human' for GUI, 'rgb_array' for array, None for headless
            opponent_policy: 'random', 'stationary', or 'simple'
            max_steps: Maximum steps per episode
            reward_shaping: If True, use shaped rewards (distance, aim, etc.)
        """
        super(TankEnv, self).__init__()
        
        self.render_mode = render_mode
        self.opponent_policy = opponent_policy
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        
        # Create game engine
        render_enabled = (render_mode == 'human')
        self.game = TankGameEngine(width=800, height=600, render=render_enabled)
        self.game.max_steps = max_steps
        
        # Define action space: Discrete with 5 actions
        self.action_space = spaces.Discrete(5)
        
        # Define observation space: Box with 26 dimensions
        # All values are normalized or in reasonable ranges
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0
        self.episode_count = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., random_positions)
        
        Returns:
            observation: Initial state observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Handle options
        random_positions = False
        if options and 'random_positions' in options:
            random_positions = options['random_positions']
        
        # Reset game engine
        state = self.game.reset(random_positions=random_positions)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0
        self.episode_count += 1
        
        # Normalize observation
        observation = self._normalize_observation(state)
        
        info = {
            'episode': self.episode_count,
            'tank1_health': self.game.tank1.health,
            'tank2_health': self.game.tank2.health
        }
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action for Tank 1 (agent's action)
        
        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode ended (win/loss)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        self.current_step += 1
        
        # Get opponent's action based on policy
        opponent_action = self._get_opponent_action()
        
        # Execute step in game engine
        state, reward_tank1, reward_tank2, done, game_info = self.game.step(
            action, opponent_action
        )
        
        # Calculate reward
        reward = self._calculate_reward(reward_tank1, action, done, game_info)
        self.episode_reward += reward
        
        # Normalize observation
        observation = self._normalize_observation(state)
        
        # Determine termination vs truncation
        terminated = done and self.current_step < self.max_steps
        truncated = self.current_step >= self.max_steps
        
        # Prepare info dict
        info = {
            'steps': self.current_step,
            'episode_reward': self.episode_reward,
            'tank1_health': self.game.tank1.health,
            'tank2_health': self.game.tank2.health,
            'winner': game_info.get('winner', None),
            'hit_opponent': reward_tank1 > 0  # True if hit this step
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            self.game.render()
            # Handle pygame events (close window, etc.)
            if not self.game.handle_events():
                # User closed window
                self.close()
    
    def close(self):
        """Clean up resources"""
        if self.game:
            self.game.close()
    
    def _normalize_observation(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize observation to reasonable ranges
        
        Args:
            state: Raw state from game engine
        
        Returns:
            Normalized state
        """
        normalized = state.copy()
        
        # Normalize positions (assuming 800x600 screen)
        # Tank 1
        normalized[0] /= 800.0  # x position
        normalized[1] /= 600.0  # y position
        normalized[2] /= 360.0  # angle
        normalized[3] /= 100.0  # health
        # normalized[4] is already 0/1 (alive)
        normalized[5] /= 3.0    # bullet count (max 3)
        normalized[6] /= 30.0   # fire cooldown (max 30)
        
        # Tank 2
        normalized[7] /= 800.0   # x position
        normalized[8] /= 600.0   # y position
        normalized[9] /= 360.0   # angle
        normalized[10] /= 100.0  # health
        # normalized[11] is already 0/1 (alive)
        normalized[12] /= 3.0    # bullet count
        normalized[13] /= 30.0   # fire cooldown
        
        # Bullets are already normalized in game engine (14-25)
        
        return normalized
    
    def _calculate_reward(
        self,
        base_reward: float,
        action: int,
        done: bool,
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for the agent
        
        Args:
            base_reward: Base reward from game engine
            action: Action taken
            done: Whether episode ended
            info: Game info dict
        
        Returns:
            Final reward
        """
        reward = base_reward
        
        if not self.reward_shaping:
            # Simple sparse rewards
            # base_reward already includes: +10 for hit, +100/-100 for win/loss
            # Add small step penalty to encourage faster wins
            if not done:
                reward -= 0.01
        else:
            # Shaped rewards (for Experiment 2)
            # Add rewards for: distance to opponent, aiming at opponent, etc.
            
            # Step penalty
            reward -= 0.01
            
            # Distance reward (closer is better)
            tank1_pos = np.array([self.game.tank1.x, self.game.tank1.y])
            tank2_pos = np.array([self.game.tank2.x, self.game.tank2.y])
            distance = np.linalg.norm(tank1_pos - tank2_pos)
            max_distance = np.sqrt(800**2 + 600**2)  # diagonal
            distance_reward = (1 - distance / max_distance) * 0.1  # Small bonus for being close
            reward += distance_reward
            
            # Aiming reward (facing opponent)
            dx = tank2_pos[0] - tank1_pos[0]
            dy = tank2_pos[1] - tank1_pos[1]
            target_angle = np.degrees(np.arctan2(dy, dx)) % 360
            current_angle = self.game.tank1.angle % 360
            angle_diff = min(abs(target_angle - current_angle), 
                           360 - abs(target_angle - current_angle))
            aiming_reward = (1 - angle_diff / 180) * 0.05  # Small bonus for aiming
            reward += aiming_reward
            
            # Survival bonus (still alive)
            if self.game.tank1.alive:
                reward += 0.02
        
        return reward
    
    def _get_opponent_action(self) -> int:
        """
        Get action for opponent tank based on policy
        
        Returns:
            Action for Tank 2 (opponent)
        """
        if self.opponent_policy == 'stationary':
            # Opponent doesn't move, only shoots occasionally
            if np.random.random() < 0.05:  # 5% chance to shoot
                return 3  # Fire
            return 4  # Do nothing
        
        elif self.opponent_policy == 'simple':
            # Simple AI: move forward and shoot
            if np.random.random() < 0.1:  # 10% chance to shoot
                return 3  # Fire
            elif np.random.random() < 0.2:  # 20% chance to turn
                return np.random.choice([1, 2])  # Random turn
            return 0  # Move forward
        
        else:  # 'random'
            # Completely random actions
            return self.action_space.sample()
    
    def set_opponent_policy(self, policy: str):
        """
        Change opponent policy during runtime
        
        Args:
            policy: 'random', 'stationary', or 'simple'
        """
        if policy in ['random', 'stationary', 'simple']:
            self.opponent_policy = policy
        else:
            raise ValueError(f"Unknown opponent policy: {policy}")


# Convenience function to create environment
def make_tank_env(
    render_mode: Optional[str] = None,
    opponent_policy: str = 'random',
    reward_shaping: bool = False
) -> TankEnv:
    """
    Factory function to create TankEnv
    
    Args:
        render_mode: 'human' or None
        opponent_policy: 'random', 'stationary', or 'simple'
        reward_shaping: Whether to use reward shaping
    
    Returns:
        TankEnv instance
    """
    return TankEnv(
        render_mode=render_mode,
        opponent_policy=opponent_policy,
        reward_shaping=reward_shaping
    )
