"""
Tank Game Engine - Core game logic for 2-player tank battle

This module contains the fundamental logic of a simple tank battle game based on pygame.
It is specifically designed for Reinforcement Learning (RL) integration, providing a clean
programmatic interface for training AI agents. The engine supports both headless (no GUI)
and rendered modes, making it suitable for both training and visualization purposes.

Key Features:
- Two-tank battle simulation with physics and collision detection
- Bullet mechanics with speed, trajectory, and hit detection
- Health system with damage calculation
- Programmatic control interface (no human input required)
- Render/headless mode support for flexible deployment
- State vector extraction for ML/RL algorithms
"""

import pygame
import numpy as np
import math
from typing import Tuple, List, Optional


class Bullet:
    """
    Bullet class representing projectiles fired by tanks
    
    Handles bullet movement, trajectory calculation, and collision detection.
    Each bullet has position, direction (angle), speed, and an active state.
    Bullets move in a straight line based on their initial firing angle.
    """
    
    def __init__(self, x: float, y: float, angle: float, speed: float = 10):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.radius = 3
        self.active = True
        
    def update(self):
        """
        Update bullet position based on velocity and direction
        
        Moves the bullet along its trajectory using trigonometric calculations.
        The bullet travels at a constant speed in the direction determined by its angle.
        """
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed
        
    def get_rect(self):
        """
        Get bounding rectangle for collision detection
        
        Returns:
            pygame.Rect: Rectangle representing the bullet's collision box,
                        centered at the bullet's position with radius dimensions
        """
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)


class Tank:
    """
    Tank class representing a player-controlled or AI-controlled tank
    
    Manages all tank properties including position, orientation, health, movement,
    rotation, and weapon systems. Each tank can move, rotate, and fire bullets with
    cooldown mechanics. Tanks have a limited number of concurrent bullets (3 max)
    and a firing cooldown to prevent spam.
    
    Attributes:
        x, y: Position coordinates on the battlefield
        angle: Orientation in degrees (0 = facing right, 90 = down, etc.)
        color: RGB color tuple for rendering
        tank_id: Unique identifier (0 or 1 typically)
        width, height: Tank dimensions for collision detection
        speed: Movement speed in pixels per frame
        rotation_speed: Rotation speed in degrees per frame
        health: Current health points (0-100)
        bullets: List of active bullets fired by this tank
        fire_cooldown: Frames remaining until can fire again
        alive: Boolean indicating if tank is still operational
    """
    
    def __init__(self, x: float, y: float, color: Tuple[int, int, int], 
                 tank_id: int = 0):
        self.x = x
        self.y = y
        self.angle = 0  # Angle in degrees (0 = facing right/east)
        self.color = color
        self.tank_id = tank_id
        
        # Tank physical properties and capabilities
        self.width = 30
        self.height = 40
        self.speed = 3  # Pixels per frame when moving
        self.rotation_speed = 5  # Degrees per frame when rotating
        self.max_health = 100
        self.health = self.max_health
        
        # Weapon system properties
        self.bullets: List[Bullet] = []
        self.max_bullets = 3  # Maximum concurrent bullets
        self.fire_cooldown = 0  # Current cooldown (0 = can fire)
        self.fire_cooldown_time = 30  # Cooldown duration in frames (~0.5 sec at 60 FPS)
        
        self.alive = True
        
    def move_forward(self):
        """
        Move tank forward in the direction it's currently facing
        
        Uses trigonometry to calculate new position based on current angle and speed.
        The tank moves along its forward vector (determined by angle).
        """
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed
        
    def move_backward(self):
        """
        Move tank backward (opposite to facing direction)
        
        Moves in the opposite direction of the tank's current orientation.
        Useful for retreating or repositioning.
        """
        self.x -= math.cos(math.radians(self.angle)) * self.speed
        self.y -= math.sin(math.radians(self.angle)) * self.speed
        
    def rotate_left(self):
        """
        Rotate tank counter-clockwise (left turn)
        
        Decreases angle by rotation_speed degrees and keeps it within 0-360 range.
        """
        self.angle -= self.rotation_speed
        self.angle %= 360
        
    def rotate_right(self):
        """
        Rotate tank clockwise (right turn)
        
        Increases angle by rotation_speed degrees and keeps it within 0-360 range.
        """
        self.angle += self.rotation_speed
        self.angle %= 360
        
    def fire(self) -> bool:
        """
        Fire a bullet from the tank's barrel
        
        Creates a new bullet if cooldown is ready and bullet limit not reached.
        The bullet spawns from the tip of the tank's barrel (front of the tank)
        and travels in the direction the tank is facing.
        
        Returns:
            bool: True if bullet was successfully fired, False if on cooldown
                  or max bullets already active
        """
        if self.fire_cooldown == 0 and len(self.bullets) < self.max_bullets:
            # Calculate bullet spawn position at barrel tip (front of tank)
            bullet_x = self.x + math.cos(math.radians(self.angle)) * (self.height / 2)
            bullet_y = self.y + math.sin(math.radians(self.angle)) * (self.height / 2)
            
            bullet = Bullet(bullet_x, bullet_y, self.angle)
            self.bullets.append(bullet)
            self.fire_cooldown = self.fire_cooldown_time
            return True
        return False
        
    def update(self, screen_width: int, screen_height: int):
        """
        Update tank state including cooldowns, position clamping, and bullet management
        
        This method should be called once per frame to:
        1. Decrement fire cooldown if active
        2. Clamp tank position to stay within screen boundaries
        3. Update all active bullets and remove those that go off-screen
        
        Args:
            screen_width: Width of the play area in pixels
            screen_height: Height of the play area in pixels
            
        Returns:
            int: Number of bullets that went off-screen (missed shots) this frame
        """
        # Decrement fire cooldown timer
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        # Clamp tank position to stay within screen boundaries
        # Ensures tank cannot move outside the playable area
        self.x = max(self.width // 2, min(screen_width - self.width // 2, self.x))
        self.y = max(self.height // 2, min(screen_height - self.height // 2, self.y))
        
        # Update all bullets and count misses (bullets going off-screen)
        missed_bullets = 0
        for bullet in self.bullets[:]:  # Iterate over copy to allow safe removal
            bullet.update()
            # Remove bullets that exit the screen boundaries (counted as misses)
            if (bullet.x < 0 or bullet.x > screen_width or 
                bullet.y < 0 or bullet.y > screen_height):
                self.bullets.remove(bullet)
                missed_bullets += 1
        
        return missed_bullets
                
    def get_rect(self) -> pygame.Rect:
        """
        Get bounding rectangle for collision detection
        
        Returns a pygame.Rect centered at the tank's position with the tank's dimensions.
        Used for detecting collisions with bullets and other game objects.
        
        Returns:
            pygame.Rect: Rectangle representing the tank's collision box
        """
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2,
                          self.width, self.height)
                          
    def take_damage(self, damage: int = 34):
        """
        Apply damage to the tank and check if it's destroyed
        
        Reduces tank's health by the damage amount. If health reaches 0 or below,
        the tank is marked as not alive. Default damage of 34 means 3 hits = destruction
        (100 health / 34 damage ≈ 3 hits).
        
        Args:
            damage: Amount of health to subtract (default: 34)
        """
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False
            
    def get_state(self) -> np.ndarray:
        """
        Get tank state as a numerical vector for ML/RL algorithms
        
        Returns a 7-element array containing all relevant tank information:
        [x_position, y_position, angle, health, alive_status, bullet_count, fire_cooldown]
        
        Returns:
            np.ndarray: Shape (7,) float32 array with tank state information
        """
        return np.array([
            self.x,
            self.y,
            self.angle,
            self.health,
            float(self.alive),
            len(self.bullets),
            self.fire_cooldown
        ], dtype=np.float32)


class TankGameEngine:
    """
    Main game engine class for tank battle simulation
    
    This class manages all game logic and provides programmatic control for RL integration.
    It handles tank creation, physics updates, collision detection, rendering, and game state
    management. The engine can run in headless mode (no GUI) for training or with rendering
    for visualization and debugging.
    
    Key responsibilities:
    - Tank and bullet lifecycle management
    - Physics updates and collision detection
    - Win/loss/draw condition checking
    - State extraction for ML algorithms
    - Optional rendering for visualization
    - Game flow control (reset, step, close)
    
    The engine uses a step-based update system compatible with RL frameworks like Gymnasium.
    """
    
    def __init__(self, width: int = 800, height: int = 600, render: bool = False):
        self.width = width
        self.height = height
        self.render_enabled = render
        
        # Pygame başlatma
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("RL Tank Battle")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        else:
            self.screen = None
            self.clock = None
            self.font = None
            
        # Game objects - tanks initialized during reset()
        self.tank1: Optional[Tank] = None
        self.tank2: Optional[Tank] = None
        
        # Game state tracking
        self.steps = 0
        self.max_steps = 1000
        self.game_over = False
        self.winner = None  # None = ongoing, 0 = draw, 1 = tank1 wins, 2 = tank2 wins
        
    def reset(self, random_positions: bool = False):
        """
        Reset the game to initial state with new tanks
        
        Reinitializes both tanks at their starting positions (either fixed or random),
        resets all game state variables, and returns the initial observation state.
        This should be called at the start of each episode.
        
        Args:
            random_positions: If True, tanks spawn at random locations within bounds.
                            If False, tanks spawn at fixed positions facing each other.
        
        Returns:
            np.ndarray: Initial game state vector (26 dimensions)
        """
        if random_positions:
            # Random spawn positions with minimum distance from edges
            x1 = np.random.randint(100, self.width - 100)
            y1 = np.random.randint(100, self.height - 100)
            x2 = np.random.randint(100, self.width - 100)
            y2 = np.random.randint(100, self.height - 100)
        else:
            # Fixed starting positions: tanks face each other from opposite sides
            x1, y1 = 100, self.height // 2
            x2, y2 = self.width - 100, self.height // 2
            
        self.tank1 = Tank(x1, y1, (0, 100, 255), tank_id=0)  # Blue tank
        self.tank1.angle = 0  # Facing right (east)
        
        self.tank2 = Tank(x2, y2, (255, 0, 100), tank_id=1)  # Red tank
        self.tank2.angle = 180  # Facing left (west)
        
        # Reset game state counters and flags
        self.steps = 0
        self.game_over = False
        self.winner = None
        
        return self.get_state()
        
    def step(self, action_tank1: int, action_tank2: int = 4) -> Tuple[np.ndarray, float, float, bool, dict]:
        """
        Advance the game by one time step with given actions for both tanks
        
        Executes one frame of game logic: applies tank actions, updates positions,
        checks collisions, calculates rewards, and determines if episode is complete.
        
        Action Space (discrete):
            0 - Move forward in facing direction
            1 - Rotate left (counter-clockwise)
            2 - Rotate right (clockwise)
            3 - Fire bullet (if cooldown ready and bullets available)
            4 - Do nothing (idle/wait)
        
        Args:
            action_tank1: Action for tank 1 (agent's tank)
            action_tank2: Action for tank 2 (opponent tank), default=4 (idle)
        
        Returns:
            tuple containing:
                - state (np.ndarray): New game state (26-dimensional vector)
                - reward_tank1 (float): Reward for tank 1
                - reward_tank2 (float): Reward for tank 2
                - done (bool): True if episode ended (win/loss/draw)
                - info (dict): Additional information (steps, winner, health, etc.)
        """
        if self.game_over:
            return self.get_state(), 0, 0, True, {'winner': self.winner}
            
        self.steps += 1
        
        # Apply actions to both tanks
        self._apply_action(self.tank1, action_tank1)
        self._apply_action(self.tank2, action_tank2)
        
        # Update tank states and get count of missed shots (bullets going off-screen)
        missed_tank1 = self.tank1.update(self.width, self.height)
        missed_tank2 = self.tank2.update(self.width, self.height)
        
        # Check for bullet-tank collisions and calculate base rewards
        reward_tank1, reward_tank2 = self._check_collisions()
        
        # Apply penalty for missing shots (encourage accurate aim)
        reward_tank1 -= missed_tank1 * 2  # -2 points per missed bullet
        reward_tank2 -= missed_tank2 * 2
        
        # Check for game termination conditions
        done = False
        if not self.tank1.alive:
            # Tank 1 destroyed - Tank 2 wins
            self.game_over = True
            self.winner = 2
            reward_tank1 = -100  # Large penalty for losing
            reward_tank2 = 100   # Large reward for winning
            done = True
        elif not self.tank2.alive:
            # Tank 2 destroyed - Tank 1 wins
            self.game_over = True
            self.winner = 1
            reward_tank1 = 100   # Large reward for winning
            reward_tank2 = -100  # Large penalty for losing
            done = True
        elif self.steps >= self.max_steps:
            # Maximum steps reached without winner - draw
            self.game_over = True
            self.winner = 0  # Draw (both survive)
            done = True
            
        info = {
            'steps': self.steps,
            'winner': self.winner,
            'tank1_health': self.tank1.health,
            'tank2_health': self.tank2.health,
            'tank1_missed': missed_tank1,
            'tank2_missed': missed_tank2
        }
        
        return self.get_state(), reward_tank1, reward_tank2, done, info
        
    def _apply_action(self, tank: Tank, action: int):
        """
        Apply an action to a specific tank
        
        Translates discrete action integers into tank method calls.
        Action 4 (do nothing) is implemented as no-op by not calling any method.
        
        Args:
            tank: Tank object to apply action to
            action: Integer action code (0-4)
        """
        if action == 0:  # Move forward
            tank.move_forward()
        elif action == 1:  # Turn left
            tank.rotate_left()
        elif action == 2:  # Turn right
            tank.rotate_right()
        elif action == 3:  # Fire bullet
            tank.fire()
        # action == 4: Do nothing (idle)
        
    def _check_collisions(self) -> Tuple[float, float]:
        """
        Check for bullet-tank collisions and apply damage
        
        Iterates through all active bullets from both tanks and checks if they
        collide with the opponent tank. On collision, applies damage to the hit tank,
        removes the bullet, and awards points to the shooter.
        
        Returns:
            tuple: (reward_tank1, reward_tank2) where each reward is +10 per hit
        """
        reward_tank1 = 0
        reward_tank2 = 0
        
        # Check if Tank1's bullets hit Tank2
        for bullet in self.tank1.bullets[:]:  # Iterate over copy for safe removal
            if self.tank2.alive and bullet.get_rect().colliderect(self.tank2.get_rect()):
                self.tank2.take_damage(34)
                self.tank1.bullets.remove(bullet)
                reward_tank1 += 10  # Hit bonus
                
        # Check if Tank2's bullets hit Tank1
        for bullet in self.tank2.bullets[:]:
            if self.tank1.alive and bullet.get_rect().colliderect(self.tank1.get_rect()):
                self.tank1.take_damage(34)
                self.tank2.bullets.remove(bullet)
                reward_tank2 += 10  # Hit bonus
                
        return reward_tank1, reward_tank2
        
    def get_state(self) -> np.ndarray:
        """
        Extract complete game state as a numerical vector for ML/RL
        
        Constructs a 26-dimensional state vector containing all observable information:
        - Tank 1 state (7 values): x, y, angle, health, alive, bullet_count, fire_cooldown
        - Tank 2 state (7 values): x, y, angle, health, alive, bullet_count, fire_cooldown  
        - Tank 1 bullets (6 values): x, y positions of up to 3 bullets (normalized 0-1)
        - Tank 2 bullets (6 values): x, y positions of up to 3 bullets (normalized 0-1)
        
        Total: 26 values (7 + 7 + 6 + 6)
        
        Returns:
            np.ndarray: Shape (26,) float32 array representing full game state
        """
        state = np.zeros(26, dtype=np.float32)
        
        # Extract tank states (positions, health, status, etc.)
        state[0:7] = self.tank1.get_state()
        state[7:14] = self.tank2.get_state()
        
        # Tank1 bullets (up to 3 bullets) - positions normalized to [0, 1]
        for i, bullet in enumerate(self.tank1.bullets[:3]):
            state[14 + i*2] = bullet.x / self.width      # Normalized x position
            state[14 + i*2 + 1] = bullet.y / self.height # Normalized y position
            
        # Tank2 bullets (up to 3 bullets) - positions normalized to [0, 1]
        for i, bullet in enumerate(self.tank2.bullets[:3]):
            state[20 + i*2] = bullet.x / self.width      # Normalized x position
            state[20 + i*2 + 1] = bullet.y / self.height # Normalized y position
            
        return state
        
    def render(self):
        """
        Render the game visually using pygame
        
        Draws all game elements including tanks, bullets, health bars, and game-over messages.
        This method should be called once per frame when visual output is desired.
        Does nothing if render mode is disabled (headless mode).
        """
        if not self.render_enabled or self.screen is None:
            return
            
        # Draw background (dark gray)
        self.screen.fill((50, 50, 50))
        
        # Draw both tanks
        self._draw_tank(self.tank1)
        self._draw_tank(self.tank2)
        
        # Draw all active bullets (colored by owner)
        for bullet in self.tank1.bullets:
            pygame.draw.circle(self.screen, self.tank1.color, 
                             (int(bullet.x), int(bullet.y)), bullet.radius)
        for bullet in self.tank2.bullets:
            pygame.draw.circle(self.screen, self.tank2.color, 
                             (int(bullet.x), int(bullet.y)), bullet.radius)
        
        # Draw health bars at screen top
        self._draw_health_bar(self.tank1, 10, 10)  # Top-left for Tank 1
        self._draw_health_bar(self.tank2, self.width - 210, 10)  # Top-right for Tank 2
        
        # Draw game over message if episode ended
        if self.game_over and self.font:
            if self.winner == 1:
                text = self.font.render("Tank 1 (Blue) Wins!", True, (255, 255, 255))
            elif self.winner == 2:
                text = self.font.render("Tank 2 (Red) Wins!", True, (255, 255, 255))
            else:
                text = self.font.render("Draw!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(60)  # Maintain 60 FPS
        
    def _draw_tank(self, tank: Tank):
        """
        Draw a tank with rotation applied
        
        Renders tank as a rotated rectangle with a barrel line.
        The tank body is drawn as a polygon with 4 corners, rotated according
        to the tank's current angle. The barrel extends from the center.
        
        Args:
            tank: Tank object to render
        """
        if not tank.alive:
            return  # Don't draw destroyed tanks
            
        # Tank body (simple rectangle) - define 4 corners relative to center
        points = [
            (-tank.width // 2, -tank.height // 2),  # Top-left
            (tank.width // 2, -tank.height // 2),   # Top-right
            (tank.width // 2, tank.height // 2),    # Bottom-right
            (-tank.width // 2, tank.height // 2)    # Bottom-left
        ]
        
        # Apply rotation transformation to all corner points
        rotated_points = []
        for px, py in points:
            angle_rad = math.radians(tank.angle)
            # 2D rotation matrix: [cos -sin; sin cos]
            new_x = px * math.cos(angle_rad) - py * math.sin(angle_rad)
            new_y = px * math.sin(angle_rad) + py * math.cos(angle_rad)
            rotated_points.append((tank.x + new_x, tank.y + new_y))
            
        pygame.draw.polygon(self.screen, tank.color, rotated_points)
        
        # Draw barrel (gun) extending from tank center
        barrel_length = tank.height // 2 + 15  # Extends beyond tank body
        barrel_end_x = tank.x + math.cos(math.radians(tank.angle)) * barrel_length
        barrel_end_y = tank.y + math.sin(math.radians(tank.angle)) * barrel_length
        pygame.draw.line(self.screen, tank.color, (tank.x, tank.y), 
                        (barrel_end_x, barrel_end_y), 4)
        
    def _draw_health_bar(self, tank: Tank, x: int, y: int):
        """
        Draw health bar for a tank
        
        Renders a horizontal bar showing the tank's current health relative to max health.
        Bar color matches tank color when alive, gray when destroyed.
        
        Args:
            tank: Tank whose health to display
            x: Screen X position for top-left of health bar
            y: Screen Y position for top-left of health bar
        """
        bar_width = 200
        bar_height = 20
        
        # Background (dark gray) - full bar width
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, bar_width, bar_height))
        
        # Foreground (colored) - proportional to current health
        health_width = int((tank.health / tank.max_health) * bar_width)
        color = tank.color if tank.alive else (100, 100, 100)  # Gray if dead
        pygame.draw.rect(self.screen, color, (x, y, health_width, bar_height))
        
        # Border (white outline)
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, bar_width, bar_height), 2)
        
    def close(self):
        """
        Clean up and close pygame resources
        
        Should be called when done with the engine to properly release pygame resources.
        """
        if self.render_enabled:
            pygame.quit()
            
    def handle_events(self) -> bool:
        """
        Process pygame events (window close, etc.)
        
        Checks for pygame events like window close button. Should be called each frame
        when rendering is enabled to allow graceful shutdown.
        
        Returns:
            bool: True to continue running, False if user requested close
        """
        if not self.render_enabled:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # User closed window
        return True
