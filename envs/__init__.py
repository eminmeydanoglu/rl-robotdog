"""
Environment package initialization
"""

from envs.robot_dog_env import RobotDogEnv
from envs.tank_env import TankEnv, make_tank_env
from envs.tank_game_engine import TankGameEngine

__all__ = [
    'RobotDogEnv',      # Legacy
    'TankEnv',          # Main RL environment
    'make_tank_env',    # Factory function
    'TankGameEngine'    # Game engine
]
