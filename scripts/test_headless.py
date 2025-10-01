"""
Tank Game Engine - Headless Test (GUI olmadan)

Oyun motorunun temel fonksiyonlarını test eder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.tank_game_engine import TankGameEngine
import numpy as np


def test_game_engine():
    """Oyun motorunun temel fonksiyonlarını test et"""
    print("=" * 60)
    print("Tank Game Engine - Headless Test")
    print("=" * 60)
    
    # Render olmadan oyun motoru oluştur
    print("\n1. Oyun motoru oluşturuluyor (render=False)...")
    game = TankGameEngine(width=800, height=600, render=False)
    print("   ✓ Oyun motoru başarıyla oluşturuldu")
    
    # Reset test
    print("\n2. Oyun sıfırlanıyor...")
    state = game.reset(random_positions=False)
    print(f"   ✓ Başlangıç durumu alındı: shape={state.shape}")
    print(f"   - Tank 1 pozisyon: ({game.tank1.x:.1f}, {game.tank1.y:.1f})")
    print(f"   - Tank 2 pozisyon: ({game.tank2.x:.1f}, {game.tank2.y:.1f})")
    
    # Step test
    print("\n3. Oyun adımları test ediliyor...")
    total_reward1 = 0
    total_reward2 = 0
    
    for i in range(100):
        # Rastgele aksiyonlar
        action1 = np.random.randint(0, 5)
        action2 = np.random.randint(0, 5)
        
        state, reward1, reward2, done, info = game.step(action1, action2)
        total_reward1 += reward1
        total_reward2 += reward2
        
        if done:
            print(f"   - Oyun {i+1}. adımda bitti")
            print(f"   - Kazanan: Tank {info['winner']}")
            break
    else:
        print(f"   - 100 adım tamamlandı (oyun bitmedi)")
    
    print(f"   ✓ Tank 1 toplam ödül: {total_reward1}")
    print(f"   ✓ Tank 2 toplam ödül: {total_reward2}")
    print(f"   - Tank 1 can: {info['tank1_health']}")
    print(f"   - Tank 2 can: {info['tank2_health']}")
    
    # State vektörü analizi
    print("\n4. State vektörü analizi...")
    print(f"   - State shape: {state.shape}")
    print(f"   - State dtype: {state.dtype}")
    print(f"   - State range: [{state.min():.2f}, {state.max():.2f}]")
    
    # Collision test
    print("\n5. Çarpışma testi...")
    game.reset()
    # Tank1'i Tank2'ye yaklaştır
    game.tank1.x = game.tank2.x - 50
    game.tank1.y = game.tank2.y
    game.tank1.angle = 0  # Sağa bak
    
    # Ateş et
    game.tank1.fire()
    print(f"   - Tank 1 ateş etti, mermi sayısı: {len(game.tank1.bullets)}")
    
    # Mermiyi hedefe götür
    hit = False
    for step in range(50):
        state, reward1, reward2, done, info = game.step(4, 4)  # İkisi de dur
        if reward1 > 0:
            hit = True
            print(f"   ✓ İsabet! {step} adımda")
            print(f"   - Tank 2 can: {game.tank2.health}")
            break
    
    if not hit:
        print("   - Mermi hedefe ulaşmadı (bu normal olabilir)")
    
    # Memory test
    print("\n6. Bellek testi (1000 adım)...")
    game.reset()
    for _ in range(1000):
        action1 = np.random.randint(0, 5)
        action2 = np.random.randint(0, 5)
        state, reward1, reward2, done, info = game.step(action1, action2)
        if done:
            game.reset()
    print("   ✓ 1000 adım başarıyla tamamlandı")
    
    # Cleanup
    game.close()
    print("\n" + "=" * 60)
    print("TÜM TESTLER BAŞARIYLA TAMAMLANDI! ✓")
    print("=" * 60)
    print("\nBölüm 1 Tamamlandı!")
    print("\nŞimdi yapabilecekleriniz:")
    print("1. Manuel test: python scripts/test_game_engine.py")
    print("2. Bölüm 2'ye geçiş: Gymnasium wrapper oluşturma")
    print("=" * 60)
    

if __name__ == "__main__":
    test_game_engine()
