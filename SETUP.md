# TreePoinTr Setup Guide

This repository contains **GCC 11+ compatibility fixes** for TreePoinTr CUDA extensions. Follow this guide to set up the environment correctly.

## 🚀 Quick Setup (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/treePoinTr.git
cd treePoinTr
```

### 2. Create Conda Environment
```bash
# Create the environment with Python 3.10
conda create -n treepointr_py310 python=3.10 -y

# Activate the environment
conda activate treepointr_py310
```

### 3. Install PyTorch with CUDA Support
```bash
# Install PyTorch 1.13.1 with CUDA 11.7 (matches the fixed configuration)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install CUDA Extensions (The Fixed Version!)
```bash
# Run the enhanced installation script with all compatibility fixes
bash install.sh
```

### 6. Verify Installation
```bash
# Run the comprehensive test script
python test_installation.py

# Or test extensions manually
python -c "
import chamfer, cubic_feature_sampling, gridding, gridding_distance, emd
print('✅ All CUDA extensions imported successfully!')
"
```

## 🔧 What's Fixed in This Repository

This repository includes critical fixes for **GCC 11+ compatibility**:

- ✅ **CUDA Version Check Bypass** - Handles PyTorch/CUDA version mismatches
- ✅ **GCC 9 Auto-Installation** - Automatic fallback to compatible compiler
- ✅ **Proper CUDA Paths** - Correct include/library directory configuration  
- ✅ **C++14 Compatibility** - Optimized compilation flags
- ✅ **Runtime Library Setup** - Automatic library path configuration

## 📋 System Requirements

- **OS**: Ubuntu 18.04+ (or compatible Linux)
- **CUDA**: 11.5+ (tested with 12.5)
- **GPU**: NVIDIA GPU with compute capability 8.6+ (RTX 30/40 series)
- **Conda/Miniconda**: Latest version

## 🔍 Troubleshooting

### If Installation Fails:

1. **Check Environment**:
   ```bash
   conda activate treepointr_py310
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
   ```

2. **Verify CUDA Installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Re-run Installation**:
   ```bash
   # Clean previous builds
   find extensions -name "build" -type d -exec rm -rf {} + 2>/dev/null
   find extensions -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null
   
   # Re-install
   bash install.sh
   ```

### For External CUDA Packages:

If you need to install additional CUDA packages (like `pointnet2_ops`), first set up the environment:

```bash
# Set up CUDA compilation environment
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDA_HOME=/usr/local/cuda-12.5
export CPATH=/usr/local/cuda-12.5/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Then install the package
pip install <external-cuda-package>
```

## 🎯 Usage

After successful installation, you can use TreePoinTr as described in the main README.md. All CUDA extensions should work seamlessly.

```python
# Example: Test the extensions
import torch
import chamfer

# Your TreePoinTr code here...
```

## 📝 Notes

- **First-time setup**: Takes 10-15 minutes (includes compilation)
- **Library paths**: Automatically configured in `~/.bashrc`
- **GCC version**: Automatically managed by conda
- **CUDA compatibility**: Handles version mismatches gracefully

## 🆘 Getting Help

If you encounter issues:

1. Check that you're in the correct conda environment: `conda activate treepointr_py310`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Review the installation logs in the generated `install_*.log` files
4. Open an issue on this repository with your error message

---

**Happy coding with TreePoinTr! 🌳✨** 