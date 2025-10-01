# Robot Dog RL Setup Instructions

## ⚠️ PyBullet Installation Issue

PyBullet requires **Microsoft Visual C++ 14.0 or greater** to compile on Windows with Python 3.10+.

## ✅ Solution Options

### Option 1: Install C++ Build Tools (Recommended for long-term)

1. **Download and Install:**
   - Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Download "Build Tools for Visual Studio"
   - Run installer and select "Desktop development with C++"
   - Wait for installation (about 6-8 GB)

2. **After installation, activate conda environment and install:**
   ```powershell
   conda activate robotdog
   pip install pybullet
   ```

### Option 2: Use Pre-compiled Conda Package (Easiest)

Try installing PyBullet from conda-forge:
```powershell
conda activate robotdog
conda install -c conda-forge pybullet
```

### Option 3: Download Pre-built Wheel (Alternative)

Visit: https://github.com/bulletphysics/bullet3/releases
Download the appropriate `.whl` file for your Python version and install:
```powershell
conda activate robotdog
pip install path\to\downloaded\pybullet-X.X.X-cpXXX-cpXXX-win_amd64.whl
```

## 📦 What's Already Installed in `robotdog` Environment

The following packages should be installed (except PyBullet):
- ✅ gymnasium
- ✅ stable-baselines3
- ✅ torch (PyTorch)
- ✅ numpy
- ✅ tensorboard
- ✅ matplotlib
- ✅ tqdm
- ✅ pyyaml
- ✅ pandas
- ✅ imageio & imageio-ffmpeg
- ❌ pybullet (needs C++ compiler OR conda-forge)

## 🚀 Quick Start (After PyBullet is installed)

1. **Activate environment:**
   ```powershell
   conda activate robotdog
   ```

2. **Test the environment:**
   ```powershell
   python -c "from envs.robot_dog_env import RobotDogEnv; print('Success!')"
   ```

3. **Start training:**
   ```powershell
   python scripts/train.py
   ```

4. **Monitor with TensorBoard:**
   ```powershell
   tensorboard --logdir logs
   ```

## 💡 Recommended: Try Option 2 First (Conda-Forge)

It's the quickest solution and doesn't require C++ Build Tools installation.

## 📁 Project Structure

```
faruk-dog/
├── envs/              # Custom Gymnasium environments
├── models/            # Trained model checkpoints
├── configs/           # Configuration files
├── scripts/           # Training and evaluation scripts
│   ├── train.py       # PPO training script
│   └── evaluate.py    # Model evaluation script
├── utils/             # Helper functions
├── notebooks/         # Jupyter notebooks
└── requirements.txt   # All dependencies
```

## ⚙️ Environment Details

- **Name:** robotdog
- **Python:** 3.10.18
- **Location:** C:\Users\eminm\miniconda3\envs\robotdog
- **Activate:** `conda activate robotdog`
- **Deactivate:** `conda deactivate`
