#!/usr/bin/env bash

echo "Installing PointNet++ with GCC 11+ compatibility fixes..."
echo "==========================================================="

# Check if we're in the correct conda environment
if [[ "$CONDA_DEFAULT_ENV" != "treepointr_py310" ]]; then
    echo "ERROR: Please activate the treepointr_py310 conda environment first!"
    echo "Run: conda activate treepointr_py310"
    echo "Then run this script again."
    exit 1
fi

# Check if torch is available
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch is not installed in this environment."
    echo "Please install PyTorch first, then run this script again."
    exit 1
fi

# Install GCC 9 if not already installed
if ! command -v gcc &> /dev/null || [[ "$(gcc --version | head -n1)" == *"11."* ]]; then
    echo "Installing GCC 9 for compatibility..."
    conda install -c conda-forge gcc=9 gxx=9 --yes
fi

# Set compiler environment variables to use conda GCC 9
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=$CONDA_PREFIX/bin:$PATH

# Disable CUDA version check for compatibility
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA="1"
export CUDA_VISIBLE_DEVICES="0"

# Allow CUDA version mismatch (PyTorch 11.7 vs CUDA 12.5)
export CUDA_FORCE_COMPILE="1"

# Set NVCC host compiler explicitly
export NVCC_PREPEND_FLAGS="-ccbin $CONDA_PREFIX/bin/gcc"

# Set additional environment variables for this compilation
export CXXFLAGS="-std=c++14"
export NVCCFLAGS="-std=c++14"

echo "Environment setup complete."
echo "Using compiler: $(which gcc)"
echo "GCC version: $(gcc --version | head -n1)"
echo ""

# Check if external_deps directory exists
if [ ! -d "external_deps" ]; then
    echo "ERROR: external_deps directory not found!"
    echo "Please make sure you're running this script from the treePoinTr root directory."
    exit 1
fi

# Check if PointNet++ repo exists
if [ ! -d "external_deps/Pointnet2_PyTorch" ]; then
    echo "ERROR: PointNet++ repository not found!"
    echo "Please run: cd external_deps && git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git"
    exit 1
fi

echo "Installing PointNet++ with fixes..."
cd external_deps/Pointnet2_PyTorch/pointnet2_ops_lib

# Clean previous builds
rm -rf build/ pointnet2_ops.egg-info/ 2>/dev/null || true

# Install with our fixed setup.py
pip install -e . 2>&1 | tee ../../../install_pointnet2.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: PointNet++ installation failed!"
    echo "Check install_pointnet2.log for details"
    exit 1
fi

echo "✓ PointNet++ installed successfully!"
echo ""

# Test the installation
echo "Testing PointNet++ installation..."
cd ../../../

python -c "
try:
    import pointnet2_ops
    print('✅ PointNet++ imported successfully!')
    print('  Available functions:')
    print('    - pointnet2_ops.ball_query')
    print('    - pointnet2_ops.knn_query')
    print('    - pointnet2_ops.grouping_operation')
    print('    - pointnet2_ops.sampling')
    print('    - pointnet2_ops.interpolation')
except ImportError as e:
    print(f'❌ PointNet++ import failed: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️  PointNet++ import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 PointNet++ installation completed successfully!"
    echo "You can now use PointNet++ operations in your TreePoinTr models."
else
    echo ""
    echo "❌ PointNet++ installation completed but import test failed."
    echo "Check the error messages above for troubleshooting."
fi

echo ""
echo "Note: PointNet++ has been installed from: external_deps/Pointnet2_PyTorch"
echo "The installation includes GCC 11+ compatibility fixes." 