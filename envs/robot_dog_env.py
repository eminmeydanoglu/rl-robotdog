"""
Robot Dog Gymnasium Environment using PyBullet

This module provides a custom Gymnasium environment for training a quadruped robot dog.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class RobotDogEnv(gym.Env):
    """
    Custom Gymnasium Environment for a quadruped robot dog.
    
    The robot must learn to walk forward while maintaining balance.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None):
        super(RobotDogEnv, self).__init__()
        
        self.render_mode = render_mode
        self.timestep = 1./240.
        
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # Set up the simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        
        # Action space: 12 joint angles (3 per leg: hip, thigh, calf)
        # Values normalized between -1 and 1
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(12,), 
            dtype=np.float32
        )
        
        # Observation space: robot state
        # Position (3), orientation (4), linear velocity (3), angular velocity (3),
        # joint positions (12), joint velocities (12)
        obs_dim = 3 + 4 + 3 + 3 + 12 + 12  # = 37
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.robot_id = None
        self.plane_id = None
        self.step_counter = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # TODO: Load your robot URDF here
        # For now, we'll use a placeholder
        # self.robot_id = p.loadURDF("path/to/your/robot.urdf", [0, 0, 0.5])
        
        # Temporary: Create a simple box as placeholder
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.15, 0.1])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.15, 0.1], rgbaColor=[0.8, 0.5, 0.2, 1])
        self.robot_id = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 0.5]
        )
        
        self.step_counter = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        # Apply actions (scale from [-1, 1] to actual joint limits)
        # TODO: Apply actions to robot joints when URDF is loaded
        
        # Step simulation
        p.stepSimulation()
        self.step_counter += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.step_counter >= 1000  # Max episode length
        
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get the current state observation."""
        # Get robot pose
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        
        # TODO: Get joint states when robot URDF is loaded
        # For now, return placeholder joint positions and velocities
        joint_positions = np.zeros(12)
        joint_velocities = np.zeros(12)
        
        observation = np.concatenate([
            pos,                    # 3
            orn,                    # 4
            linear_vel,             # 3
            angular_vel,            # 3
            joint_positions,        # 12
            joint_velocities        # 12
        ]).astype(np.float32)
        
        return observation
    
    def _compute_reward(self):
        """Compute the reward for the current state."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, _ = p.getBaseVelocity(self.robot_id)
        
        # Reward forward movement
        forward_reward = linear_vel[0]  # x-direction velocity
        
        # Penalty for falling (low height)
        height_penalty = -10.0 if pos[2] < 0.2 else 0.0
        
        # Penalty for tilting
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        orientation_penalty = -0.1 * (abs(roll) + abs(pitch))
        
        # Energy penalty (encourage efficient movement)
        # energy_penalty = -0.001 * np.sum(np.square(action))
        
        reward = forward_reward + height_penalty + orientation_penalty
        
        return reward
    
    def _is_terminated(self):
        """Check if the episode should terminate."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Terminate if robot falls
        if pos[2] < 0.15:
            return True
            
        # Terminate if robot tilts too much
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        if abs(roll) > np.pi/3 or abs(pitch) > np.pi/3:
            return True
            
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # TODO: Implement camera rendering
            pass
        # GUI rendering is handled by PyBullet automatically
    
    def close(self):
        """Clean up resources."""
        p.disconnect()


# Register the environment
gym.register(
    id='RobotDog-v0',
    entry_point='envs.robot_dog_env:RobotDogEnv',
    max_episode_steps=1000,
)
