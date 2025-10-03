"""
Tank Game Engine Test Script - Manual Control Test

Tests that the game engine works correctly with human input.
This is a 2-player local multiplayer test where you can control both tanks manually.

Controls:
Tank 1 (Blue):
- W: Move forward
- A: Turn left
- D: Turn right
- Space: Fire bullet

Tank 2 (Red):
- Up Arrow: Move forward
- Left Arrow: Turn left
- Right Arrow: Turn right
- Enter: Fire bullet

General:
- ESC: Exit game

This script is useful for:
- Verifying game mechanics work correctly
- Testing collision detection visually
- Demonstrating the game to others
- Debugging rendering issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from envs.tank_game_engine import TankGameEngine


def main():
    """
    Main function for manual testing of the game engine
    
    Creates a game instance with rendering enabled and enters the main game loop.
    Processes keyboard input for both tanks and displays the game visually.
    """
    game = TankGameEngine(width=800, height=600, render=True)
    game.reset(random_positions=False)
    
    print("Tank Game Engine Test")
    print("=" * 50)
    print("\nControls:")
    print("\nTank 1 (Blue):")
    print("  W: Move forward")
    print("  A: Turn left")
    print("  D: Turn right")
    print("  Space: Fire")
    print("\nTank 2 (Red):")
    print("  Up Arrow: Move forward")
    print("  Left Arrow: Turn left")
    print("  Right Arrow: Turn right")
    print("  Enter: Fire")
    print("\nESC: Exit")
    print("=" * 50)
    
    running = True
    
    while running:
        # Initialize actions to idle (do nothing) each frame
        action_tank1 = 4  # Idle
        action_tank2 = 4  # Idle
        
        # Handle pygame events (window close, ESC key)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get current keyboard state for continuous input
        keys = pygame.key.get_pressed()
        
        # Tank 1 controls (WASD + Space)
        if keys[pygame.K_w]:
            action_tank1 = 0  # Forward
        elif keys[pygame.K_a]:
            action_tank1 = 1  # Turn left
        elif keys[pygame.K_d]:
            action_tank1 = 2  # Turn right
        elif keys[pygame.K_SPACE]:
            action_tank1 = 3  # Fire
            
        # Tank 2 controls (Arrow keys + Enter)
        if keys[pygame.K_UP]:
            action_tank2 = 0  # Forward
        elif keys[pygame.K_LEFT]:
            action_tank2 = 1  # Turn left
        elif keys[pygame.K_RIGHT]:
            action_tank2 = 2  # Turn right
        elif keys[pygame.K_RETURN]:
            action_tank2 = 3  # Fire
        
        # Update game state with both player actions
        state, reward1, reward2, done, info = game.step(action_tank1, action_tank2)
        
        # Render the current frame
        game.render()
        
        # Check if episode ended and display results
        if done:
            print(f"\nGame Over!")
            if info['winner'] == 1:
                print("Winner: Tank 1 (Blue)")
            elif info['winner'] == 2:
                print("Winner: Tank 2 (Red)")
            else:
                print("Draw")
            print(f"Total steps: {info['steps']}")
            print(f"Tank 1 Health: {info['tank1_health']}")
            print(f"Tank 2 Health: {info['tank2_health']}")
            
            # Wait 3 seconds before starting new game
            pygame.time.wait(3000)
            
            # Start new game
            game.reset(random_positions=False)
    
    game.close()
    print("\nTest completed!")


if __name__ == "__main__":
    main()
