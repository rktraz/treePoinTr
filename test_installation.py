#!/usr/bin/env python3
"""
TreePoinTr Installation Test Script

This script verifies that all CUDA extensions are properly installed
and can be imported without errors.

Usage:
    python test_installation.py
"""

import sys
import torch
import warnings

def test_pytorch_cuda():
    """Test PyTorch and CUDA availability"""
    print("🔍 Testing PyTorch and CUDA...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU devices: {torch.cuda.device_count()}")
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Test basic GPU operations
        device = torch.device('cuda:0')
        x = torch.randn(100, 3, device=device)
        y = torch.randn(100, 3, device=device)
        z = x + y
        print(f"  GPU operations: ✅ Working")
    else:
        print("  ⚠️  CUDA not available - extensions may not work")
    
    print()

def test_cuda_extensions():
    """Test all CUDA extensions"""
    print("🚀 Testing CUDA extensions...")
    
    extensions = [
        ("chamfer", "Chamfer Distance"),
        ("cubic_feature_sampling", "Cubic Feature Sampling"), 
        ("gridding", "Gridding"),
        ("gridding_distance", "Gridding Distance"),
        ("emd", "Earth Mover's Distance")
    ]
    
    success_count = 0
    total_count = len(extensions)
    
    for module_name, display_name in extensions:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name} - OK")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {display_name} - FAILED: {e}")
        except Exception as e:
            print(f"  ⚠️  {display_name} - ERROR: {e}")
    
    print(f"\n📊 Extension Summary: {success_count}/{total_count} extensions working")
    return success_count == total_count

def test_basic_functionality():
    """Test basic functionality of extensions"""
    print("🧪 Testing basic functionality...")
    
    if not torch.cuda.is_available():
        print("  ⚠️  Skipping functionality tests - CUDA not available")
        return True
    
    try:
        # Test chamfer distance
        import chamfer
        device = torch.device('cuda:0')
        
        # Create sample point clouds
        pc1 = torch.randn(1, 1000, 3, device=device)
        pc2 = torch.randn(1, 1000, 3, device=device)
        
        # Test chamfer distance calculation
        dist1, dist2 = chamfer.chamfer_distance(pc1, pc2)
        
        print(f"  ✅ Chamfer distance computation - OK")
        print(f"    Sample distance: {dist1.mean().item():.6f}")
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🌳 TreePoinTr Installation Test")
    print("=" * 40)
    
    # Test PyTorch and CUDA
    test_pytorch_cuda()
    
    # Test CUDA extensions
    extensions_ok = test_cuda_extensions()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 40)
    
    if extensions_ok and functionality_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ TreePoinTr installation is working correctly")
        print("\nYou can now use TreePoinTr for point cloud completion!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Please check the error messages above")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the 'treepointr_py310' conda environment")
        print("2. Try running: bash install.sh")
        print("3. Check that CUDA drivers are properly installed")
        print("4. Verify PyTorch was installed with CUDA support")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 