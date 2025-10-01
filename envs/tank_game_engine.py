"""
Tank Game Engine - Core game logic for 2-player tank battle

Bu modül, pygame tabanlı basit bir tank savaş oyununun temel mantığını içerir.
RL entegrasyonu için tasarlanmıştır.
"""

import pygame
import numpy as np
import math
from typing import Tuple, List, Optional


class Bullet:
    """Mermi sınıfı"""
    
    def __init__(self, x: float, y: float, angle: float, speed: float = 10):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.radius = 3
        self.active = True
        
    def update(self):
        """Mermiyi hareket ettir"""
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed
        
    def get_rect(self):
        """Çarpışma kontrolü için rect döndür"""
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)


class Tank:
    """Tank sınıfı"""
    
    def __init__(self, x: float, y: float, color: Tuple[int, int, int], 
                 tank_id: int = 0):
        self.x = x
        self.y = y
        self.angle = 0  # Derece cinsinden
        self.color = color
        self.tank_id = tank_id
        
        # Tank özellikleri
        self.width = 30
        self.height = 40
        self.speed = 3
        self.rotation_speed = 5
        self.max_health = 100
        self.health = self.max_health
        
        # Ateş etme
        self.bullets: List[Bullet] = []
        self.max_bullets = 3
        self.fire_cooldown = 0
        self.fire_cooldown_time = 30  # frames
        
        self.alive = True
        
    def move_forward(self):
        """İleri hareket et"""
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed
        
    def move_backward(self):
        """Geri hareket et"""
        self.x -= math.cos(math.radians(self.angle)) * self.speed
        self.y -= math.sin(math.radians(self.angle)) * self.speed
        
    def rotate_left(self):
        """Sola dön"""
        self.angle -= self.rotation_speed
        self.angle %= 360
        
    def rotate_right(self):
        """Sağa dön"""
        self.angle += self.rotation_speed
        self.angle %= 360
        
    def fire(self) -> bool:
        """Ateş et"""
        if self.fire_cooldown == 0 and len(self.bullets) < self.max_bullets:
            # Namlu ucundan ateş et
            bullet_x = self.x + math.cos(math.radians(self.angle)) * (self.height / 2)
            bullet_y = self.y + math.sin(math.radians(self.angle)) * (self.height / 2)
            
            bullet = Bullet(bullet_x, bullet_y, self.angle)
            self.bullets.append(bullet)
            self.fire_cooldown = self.fire_cooldown_time
            return True
        return False
        
    def update(self, screen_width: int, screen_height: int):
        """Tank durumunu güncelle"""
        # Cooldown
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        # Sınırları kontrol et
        self.x = max(self.width // 2, min(screen_width - self.width // 2, self.x))
        self.y = max(self.height // 2, min(screen_height - self.height // 2, self.y))
        
        # Mermileri güncelle
        missed_bullets = 0
        for bullet in self.bullets[:]:
            bullet.update()
            # Ekran dışına çıkan mermileri kaldır (miss)
            if (bullet.x < 0 or bullet.x > screen_width or 
                bullet.y < 0 or bullet.y > screen_height):
                self.bullets.remove(bullet)
                missed_bullets += 1
        
        return missed_bullets
                
    def get_rect(self) -> pygame.Rect:
        """Çarpışma kontrolü için rect döndür"""
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2,
                          self.width, self.height)
                          
    def take_damage(self, damage: int = 34):
        """Hasar al"""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False
            
    def get_state(self) -> np.ndarray:
        """Tank durumunu vektör olarak döndür"""
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
    Tank oyununun ana motor sınıfı
    
    Bu sınıf oyunun tüm mantığını yönetir ve RL entegrasyonu için
    programatik kontrol sağlar.
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
            
        # Oyun nesneleri
        self.tank1: Optional[Tank] = None
        self.tank2: Optional[Tank] = None
        
        self.steps = 0
        self.max_steps = 1000
        self.game_over = False
        self.winner = None
        
    def reset(self, random_positions: bool = False):
        """Oyunu başlangıç durumuna getir"""
        if random_positions:
            # Rastgele pozisyonlar
            x1 = np.random.randint(100, self.width - 100)
            y1 = np.random.randint(100, self.height - 100)
            x2 = np.random.randint(100, self.width - 100)
            y2 = np.random.randint(100, self.height - 100)
        else:
            # Sabit başlangıç pozisyonları
            x1, y1 = 100, self.height // 2
            x2, y2 = self.width - 100, self.height // 2
            
        self.tank1 = Tank(x1, y1, (0, 100, 255), tank_id=0)  # Mavi
        self.tank1.angle = 0
        
        self.tank2 = Tank(x2, y2, (255, 0, 100), tank_id=1)  # Kırmızı
        self.tank2.angle = 180
        
        self.steps = 0
        self.game_over = False
        self.winner = None
        
        return self.get_state()
        
    def step(self, action_tank1: int, action_tank2: int = 4) -> Tuple[np.ndarray, float, float, bool, dict]:
        """
        Oyunu bir adım ilerlet
        
        Actions:
        0 - İleri git
        1 - Sola dön
        2 - Sağa dön
        3 - Ateş et
        4 - Dur (hiçbir şey yapma)
        
        Returns:
            state, reward_tank1, reward_tank2, done, info
        """
        if self.game_over:
            return self.get_state(), 0, 0, True, {'winner': self.winner}
            
        self.steps += 1
        
        # Tank1 aksiyonunu uygula
        self._apply_action(self.tank1, action_tank1)
        
        # Tank2 aksiyonunu uygula
        self._apply_action(self.tank2, action_tank2)
        
        # Tankları güncelle ve miss sayısını al
        missed_tank1 = self.tank1.update(self.width, self.height)
        missed_tank2 = self.tank2.update(self.width, self.height)
        
        # Çarpışmaları kontrol et
        reward_tank1, reward_tank2 = self._check_collisions()
        
        # Miss penalty ekle
        reward_tank1 -= missed_tank1 * 2  # Her miss için -2 puan
        reward_tank2 -= missed_tank2 * 2
        
        # Oyun bitişini kontrol et
        done = False
        if not self.tank1.alive:
            self.game_over = True
            self.winner = 2
            reward_tank1 = -100
            reward_tank2 = 100
            done = True
        elif not self.tank2.alive:
            self.game_over = True
            self.winner = 1
            reward_tank1 = 100
            reward_tank2 = -100
            done = True
        elif self.steps >= self.max_steps:
            self.game_over = True
            self.winner = 0  # Berabere
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
        """Bir tanka aksiyon uygula"""
        if action == 0:  # İleri
            tank.move_forward()
        elif action == 1:  # Sola dön
            tank.rotate_left()
        elif action == 2:  # Sağa dön
            tank.rotate_right()
        elif action == 3:  # Ateş et
            tank.fire()
        # action == 4: Dur (hiçbir şey yapma)
        
    def _check_collisions(self) -> Tuple[float, float]:
        """Mermi çarpışmalarını kontrol et ve ödülleri döndür"""
        reward_tank1 = 0
        reward_tank2 = 0
        
        # Tank1'in mermileri Tank2'ye çarptı mı?
        for bullet in self.tank1.bullets[:]:
            if self.tank2.alive and bullet.get_rect().colliderect(self.tank2.get_rect()):
                self.tank2.take_damage(34)
                self.tank1.bullets.remove(bullet)
                reward_tank1 += 10  # İsabet bonusu
                
        # Tank2'nin mermileri Tank1'e çarptı mı?
        for bullet in self.tank2.bullets[:]:
            if self.tank1.alive and bullet.get_rect().colliderect(self.tank1.get_rect()):
                self.tank1.take_damage(34)
                self.tank2.bullets.remove(bullet)
                reward_tank2 += 10  # İsabet bonusu
                
        return reward_tank1, reward_tank2
        
    def get_state(self) -> np.ndarray:
        """
        Oyun durumunu vektör olarak döndür
        
        State vector:
        - Tank1 durumu (7 değer)
        - Tank2 durumu (7 değer)
        - Tank1'in mermileri (3 * 2 = 6 değer: x, y pozisyonları)
        - Tank2'nin mermileri (3 * 2 = 6 değer: x, y pozisyonları)
        Toplam: 26 değer
        """
        state = np.zeros(26, dtype=np.float32)
        
        # Tank durumları
        state[0:7] = self.tank1.get_state()
        state[7:14] = self.tank2.get_state()
        
        # Tank1 mermileri (ilk 3 mermi)
        for i, bullet in enumerate(self.tank1.bullets[:3]):
            state[14 + i*2] = bullet.x / self.width  # Normalize
            state[14 + i*2 + 1] = bullet.y / self.height
            
        # Tank2 mermileri (ilk 3 mermi)
        for i, bullet in enumerate(self.tank2.bullets[:3]):
            state[20 + i*2] = bullet.x / self.width  # Normalize
            state[20 + i*2 + 1] = bullet.y / self.height
            
        return state
        
    def render(self):
        """Oyunu görselleştir"""
        if not self.render_enabled or self.screen is None:
            return
            
        # Arka plan
        self.screen.fill((50, 50, 50))
        
        # Tankları çiz
        self._draw_tank(self.tank1)
        self._draw_tank(self.tank2)
        
        # Mermileri çiz
        for bullet in self.tank1.bullets:
            pygame.draw.circle(self.screen, self.tank1.color, 
                             (int(bullet.x), int(bullet.y)), bullet.radius)
        for bullet in self.tank2.bullets:
            pygame.draw.circle(self.screen, self.tank2.color, 
                             (int(bullet.x), int(bullet.y)), bullet.radius)
        
        # Can barlarını çiz
        self._draw_health_bar(self.tank1, 10, 10)
        self._draw_health_bar(self.tank2, self.width - 210, 10)
        
        # Oyun bitti mesajı
        if self.game_over and self.font:
            if self.winner == 1:
                text = self.font.render("Tank 1 (Mavi) Kazandı!", True, (255, 255, 255))
            elif self.winner == 2:
                text = self.font.render("Tank 2 (Kırmızı) Kazandı!", True, (255, 255, 255))
            else:
                text = self.font.render("Berabere!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(60)
        
    def _draw_tank(self, tank: Tank):
        """Tankı çiz"""
        if not tank.alive:
            return
            
        # Tank gövdesi (basit dörtgen)
        points = [
            (-tank.width // 2, -tank.height // 2),
            (tank.width // 2, -tank.height // 2),
            (tank.width // 2, tank.height // 2),
            (-tank.width // 2, tank.height // 2)
        ]
        
        # Rotasyonu uygula
        rotated_points = []
        for px, py in points:
            angle_rad = math.radians(tank.angle)
            new_x = px * math.cos(angle_rad) - py * math.sin(angle_rad)
            new_y = px * math.sin(angle_rad) + py * math.cos(angle_rad)
            rotated_points.append((tank.x + new_x, tank.y + new_y))
            
        pygame.draw.polygon(self.screen, tank.color, rotated_points)
        
        # Namlu
        barrel_length = tank.height // 2 + 15
        barrel_end_x = tank.x + math.cos(math.radians(tank.angle)) * barrel_length
        barrel_end_y = tank.y + math.sin(math.radians(tank.angle)) * barrel_length
        pygame.draw.line(self.screen, tank.color, (tank.x, tank.y), 
                        (barrel_end_x, barrel_end_y), 4)
        
    def _draw_health_bar(self, tank: Tank, x: int, y: int):
        """Can barını çiz"""
        bar_width = 200
        bar_height = 20
        
        # Arka plan
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, bar_width, bar_height))
        
        # Can barı
        health_width = int((tank.health / tank.max_health) * bar_width)
        color = tank.color if tank.alive else (100, 100, 100)
        pygame.draw.rect(self.screen, color, (x, y, health_width, bar_height))
        
        # Çerçeve
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, bar_width, bar_height), 2)
        
    def close(self):
        """Pygame'i kapat"""
        if self.render_enabled:
            pygame.quit()
            
    def handle_events(self) -> bool:
        """Pygame eventlerini işle (kapatma vs.)"""
        if not self.render_enabled:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
