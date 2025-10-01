# RL Tank Game - Kurulum Talimatları

Bu döküman, projeyi sıfırdan kurmak için gereken tüm adımları içermektedir.

## Ön Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)
- Git (opsiyonel, ancak önerilir)

## Adım 1: Projeyi İndirin

```bash
git clone <repository-url>
cd faruk-dog
```

## Adım 2: Sanal Ortam Oluşturun

### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Adım 3: Kütüphaneleri Yükleyin

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Adım 4: Kurulumu Test Edin

### Oyun Motorunu Test Edin (Manuel Oyun)

```bash
python scripts/test_game_engine.py
```

**Kontroller:**
- Tank 1 (Mavi): W (ileri), A (sola), D (sağa), Space (ateş)
- Tank 2 (Kırmızı): Yukarı ok (ileri), Sol ok (sola), Sağ ok (sağa), Enter (ateş)
- ESC: Çıkış

Oyun penceresi açılmalı ve her iki tankı da kontrol edebilmelisiniz.

## Proje Yapısı

```
faruk-dog/
├── configs/              # Yapılandırma dosyaları
│   └── train_config.yaml
├── envs/                # Gymnasium ortamları
│   ├── __init__.py
│   ├── tank_game_engine.py   # Oyun motoru (YENİ!)
│   └── robot_dog_env.py      # Eski ortam (kaldırılacak)
├── models/              # Eğitilmiş modeller buraya kaydedilir
├── notebooks/           # Jupyter notebook'lar (analiz için)
├── scripts/             # Eğitim ve test scriptleri
│   ├── train.py
│   ├── evaluate.py
│   └── test_game_engine.py   # Test scripti (YENİ!)
├── utils/               # Yardımcı fonksiyonlar
│   ├── __init__.py
│   └── helpers.py
├── requirements.txt     # Python bağımlılıkları
├── README.md
└── SETUP_INSTRUCTIONS.md
```

## Sık Karşılaşılan Sorunlar

### Pygame Kurulamıyor

Windows'ta pygame kurulum hatası alırsanız:
```bash
pip install pygame --pre
```

### Sanal Ortam Aktifleştirilemiyor (Windows)

PowerShell execution policy hatası:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ImportError: No module named 'envs'

Script'leri proje kök dizininden çalıştırdığınızdan emin olun:
```bash
cd c:\Users\eminm\faruk-dog
python scripts/test_game_engine.py
```

## Sonraki Adımlar

Kurulum başarılı olduysa, şimdi şunları yapabilirsiniz:

1. ✅ Bölüm 1: Tamamlandı - Ortam kuruldu ve oyun motoru hazır
2. 🔄 Bölüm 2: Gymnasium wrapper'ı oluşturma (devam edecek)
3. ⏭️ Bölüm 3: Tek ajanlı eğitim
4. ⏭️ Bölüm 4: Multi-agent eğitim ve karşılaştırmalar

## Yardım

Sorun yaşarsanız:
1. Python versiyonunuzu kontrol edin: `python --version`
2. Sanal ortamın aktif olduğundan emin olun
3. Bağımlılıkları yeniden yükleyin: `pip install -r requirements.txt --force-reinstall`
- **Activate:** `conda activate robotdog`
- **Deactivate:** `conda deactivate`
