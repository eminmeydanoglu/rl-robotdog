"""
Utility functions for the robot dog project
"""

import numpy as np
import yaml


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def quaternion_to_euler(quaternion):
    """
    Convert quaternion to euler angles (roll, pitch, yaw).
    
    Args:
        quaternion: [x, y, z, w]
    
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    x, y, z, w = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class MovingAverage:
    """Calculate moving average of values."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = []
    
    def add(self, value):
        """Add a value to the moving average."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get(self):
        """Get the current moving average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        """Reset the moving average."""
        self.values = []
