"""
Tank Game Engine Test Script

Oyun motorunun doğru çalıştığını test eder.
İki tankı manuel kontrol edebilirsiniz:

Tank 1 (Mavi):
- W: İleri
- A: Sola dön
- D: Sağa dön
- Space: Ateş et

Tank 2 (Kırmızı):
- Yukarı Ok: İleri
- Sol Ok: Sola dön
- Sağ Ok: Sağa dön
- Enter: Ateş et
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from envs.tank_game_engine import TankGameEngine


def main():
    """Manuel test için ana fonksiyon"""
    game = TankGameEngine(width=800, height=600, render=True)
    game.reset(random_positions=False)
    
    print("Tank Game Engine Test")
    print("=" * 50)
    print("\nKontroller:")
    print("\nTank 1 (Mavi):")
    print("  W: İleri")
    print("  A: Sola dön")
    print("  D: Sağa dön")
    print("  Space: Ateş et")
    print("\nTank 2 (Kırmızı):")
    print("  Yukarı Ok: İleri")
    print("  Sol Ok: Sola dön")
    print("  Sağ Ok: Sağa dön")
    print("  Enter: Ateş et")
    print("\nESC: Çıkış")
    print("=" * 50)
    
    running = True
    
    while running:
        # Event handling
        action_tank1 = 4  # Dur
        action_tank2 = 4  # Dur
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Klavye durumunu al
        keys = pygame.key.get_pressed()
        
        # Tank 1 kontrolleri
        if keys[pygame.K_w]:
            action_tank1 = 0  # İleri
        elif keys[pygame.K_a]:
            action_tank1 = 1  # Sola
        elif keys[pygame.K_d]:
            action_tank1 = 2  # Sağa
        elif keys[pygame.K_SPACE]:
            action_tank1 = 3  # Ateş
            
        # Tank 2 kontrolleri
        if keys[pygame.K_UP]:
            action_tank2 = 0  # İleri
        elif keys[pygame.K_LEFT]:
            action_tank2 = 1  # Sola
        elif keys[pygame.K_RIGHT]:
            action_tank2 = 2  # Sağa
        elif keys[pygame.K_RETURN]:
            action_tank2 = 3  # Ateş
        
        # Oyunu güncelle
        state, reward1, reward2, done, info = game.step(action_tank1, action_tank2)
        
        # Render
        game.render()
        
        # Oyun bitti mi?
        if done:
            print(f"\nOyun Bitti!")
            print(f"Kazanan: Tank {info['winner']}" if info['winner'] > 0 else "Berabere")
            print(f"Toplam adım: {info['steps']}")
            print(f"Tank 1 Can: {info['tank1_health']}")
            print(f"Tank 2 Can: {info['tank2_health']}")
            
            # 3 saniye bekle
            pygame.time.wait(3000)
            
            # Yeni oyun
            game.reset(random_positions=False)
    
    game.close()
    print("\nTest tamamlandı!")


if __name__ == "__main__":
    main()
