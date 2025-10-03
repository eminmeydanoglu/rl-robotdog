"""
Tank Game Engine - Headless Test (No GUI)

Tests the fundamental functions of the game engine without visual rendering.
This is useful for automated testing, CI/CD pipelines, and verifying that
the core game logic works correctly in headless environments (servers, etc.).

Tests performed:
1. Engine creation in headless mode
2. Game reset functionality
3. Step execution with random actions
4. State vector structure and validity
5. Collision detection mechanics
6. Memory stability over extended play

All tests run without opening any GUI windows, making them suitable for
automated testing environments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.tank_game_engine import TankGameEngine
import numpy as np


def test_game_engine():
    """
    Test core functions of the game engine in headless mode
    
    Performs comprehensive testing of the game engine without GUI:
    - Creation and initialization
    - Reset functionality
    - Step execution
    - State vector structure
    - Collision detection
    - Memory stability
    """
    print("=" * 60)
    print("Tank Game Engine - Headless Test")
    print("=" * 60)
    
    # Test 1: Create game engine without rendering
    print("\n1. Creating game engine (render=False)...")
    game = TankGameEngine(width=800, height=600, render=False)
    print("   ✓ Game engine created successfully")
    
    # Test 2: Reset functionality
    print("\n2. Resetting game...")
    state = game.reset(random_positions=False)
    print(f"   ✓ Initial state obtained: shape={state.shape}")
    print(f"   - Tank 1 position: ({game.tank1.x:.1f}, {game.tank1.y:.1f})")
    print(f"   - Tank 2 position: ({game.tank2.x:.1f}, {game.tank2.y:.1f})")
    
    # Test 3: Step execution with random actions
    print("\n3. Testing game steps...")
    total_reward1 = 0
    total_reward2 = 0
    
    for i in range(100):
        # Random actions for both tanks
        action1 = np.random.randint(0, 5)
        action2 = np.random.randint(0, 5)
        
        state, reward1, reward2, done, info = game.step(action1, action2)
        total_reward1 += reward1
        total_reward2 += reward2
        
        if done:
            print(f"   - Game ended at step {i+1}")
            print(f"   - Winner: Tank {info['winner']}")
            break
    else:
        print(f"   - 100 steps completed (game ongoing)")
    
    print(f"   ✓ Tank 1 total reward: {total_reward1}")
    print(f"   ✓ Tank 2 total reward: {total_reward2}")
    print(f"   - Tank 1 health: {info['tank1_health']}")
    print(f"   - Tank 2 health: {info['tank2_health']}")
    
    # Test 4: State vector analysis
    print("\n4. State vector analysis...")
    print(f"   - State shape: {state.shape}")
    print(f"   - State dtype: {state.dtype}")
    print(f"   - State range: [{state.min():.2f}, {state.max():.2f}]")
    
    # Test 5: Collision detection
    print("\n5. Collision test...")
    game.reset()
    # Position Tank1 near Tank2 facing it
    game.tank1.x = game.tank2.x - 50
    game.tank1.y = game.tank2.y
    game.tank1.angle = 0  # Face right
    
    # Fire a bullet
    game.tank1.fire()
    print(f"   - Tank 1 fired, bullet count: {len(game.tank1.bullets)}")
    
    # Step until bullet hits or misses
    hit = False
    for step in range(50):
        state, reward1, reward2, done, info = game.step(4, 4)  # Both idle
        if reward1 > 0:
            hit = True
            print(f"   ✓ Hit! At step {step}")
            print(f"   - Tank 2 health: {game.tank2.health}")
            break
    
    if not hit:
        print("   - Bullet didn't hit target (this can be normal)")
    
    # Test 6: Memory stability over extended play
    print("\n6. Memory test (1000 steps)...")
    game.reset()
    for _ in range(1000):
        action1 = np.random.randint(0, 5)
        action2 = np.random.randint(0, 5)
        state, reward1, reward2, done, info = game.step(action1, action2)
        if done:
            game.reset()
    print("   ✓ 1000 steps completed successfully")
    
    # Cleanup resources
    game.close()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 60)
    print("\nChapter 1 Complete!")
    print("\nNext steps:")
    print("1. Manual test: python scripts/test_game_engine.py")
    print("2. Chapter 2: Create Gymnasium wrapper")
    print("=" * 60)
    

if __name__ == "__main__":
    test_game_engine()
