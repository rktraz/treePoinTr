#!/usr/bin/env bash
HOME=`pwd`

echo "Installing TreePoinTr CUDA extensions..."
echo "Using GCC 9 and C++14 standard for compatibility"

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

# Verify we're using the correct compiler
echo "Using compiler: $(which gcc)"
echo "GCC version: $(gcc --version | head -n1)"

# Set NVCC host compiler explicitly
export NVCC_PREPEND_FLAGS="-ccbin $CONDA_PREFIX/bin/gcc"

echo "Environment check passed. Proceeding with installation..."

# Clean previous builds
echo "Cleaning previous builds..."
find extensions -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
find extensions -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Function to install with error handling
install_extension() {
    local extension_name=$1
    local extension_path=$2
    
    echo "Installing $extension_name..."
    cd $HOME/$extension_path
    
    # Set additional environment variables for this compilation
    export CXXFLAGS="-std=c++14"
    export NVCCFLAGS="-std=c++14"
    
    pip install -e . 2>&1 | tee install_$extension_name.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: $extension_name installation failed!"
        echo "Check install_$extension_name.log for details"
        return 1
    fi
    echo "✓ $extension_name installed successfully"
    return 0
}

# Install each extension
install_extension "Chamfer Distance" "extensions/chamfer_dist" || exit 1
install_extension "Cubic Feature Sampling" "extensions/cubic_feature_sampling" || exit 1
install_extension "Gridding" "extensions/gridding" || exit 1
install_extension "Gridding Loss" "extensions/gridding_loss" || exit 1
install_extension "EMD" "extensions/emd" || exit 1

echo "✓ Installation complete!"
echo "All CUDA extensions have been successfully installed."
echo ""
echo "Setting up library paths..."
echo "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "Library path configuration added to ~/.bashrc"
echo ""
echo "To use the extensions immediately, run:"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Note: If you encounter any issues, try installing ninja for faster compilation:"
echo "  conda install ninja"
echo "  or"
echo "  pip install ninja"

