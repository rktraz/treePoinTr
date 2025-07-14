#!/usr/bin/env python3
"""
Complete Tree Processing Script
Takes a PLY file, cuts it into cubes, runs TreePoinTr inference, and concatenates results.
Usage: python complete_tree.py input_tree.ply
"""

import argparse
import os
import glob
import numpy as np
import open3d as o3d
import subprocess
import sys
from datetime import datetime
import shutil

# Add tree2cubes import
sys.path.append('tree_workflow')
sys.path.append('.')  # Add current directory for tree2cubes_improved
try:
    import tree2cubes
    import tree2cubes_improved
except ImportError:
    print("Error: tree2cubes module not found in tree_workflow/")
    sys.exit(1)

def create_run_folder():
    """Create timestamped run folder structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"inference_runs/{timestamp}_run"
    cubes_folder = os.path.join(run_folder, "cubes")
    
    os.makedirs(cubes_folder, exist_ok=True)
    return run_folder, cubes_folder

def cut_point_cloud_to_cubes(point_cloud, cubes_folder, cube_sizes, enable_flipping=True, 
                           use_improved=True, min_points=100, max_points=8192, 
                           target_points=3000, debug_analysis=True):
    """Cut point cloud into cubes and save as .txt files"""
    print("🔪 Cutting point cloud into cubes...")
    
    if debug_analysis:
        print(f"📊 Input tree analysis:")
        print(f"   Total points: {point_cloud.shape[0]:,}")
        print(f"   X range: {np.min(point_cloud[:, 0]):.2f} to {np.max(point_cloud[:, 0]):.2f} ({np.max(point_cloud[:, 0]) - np.min(point_cloud[:, 0]):.2f}m)")
        print(f"   Y range: {np.min(point_cloud[:, 1]):.2f} to {np.max(point_cloud[:, 1]):.2f} ({np.max(point_cloud[:, 1]) - np.min(point_cloud[:, 1]):.2f}m)")
        print(f"   Z range: {np.min(point_cloud[:, 2]):.2f} to {np.max(point_cloud[:, 2]):.2f} ({np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2]):.2f}m)")
        print(f"   Cube sizes: {cube_sizes}")
    
    # Cut into different cube sizes using configured method
    if use_improved and hasattr(tree2cubes_improved, 'cut_point_cloud_improved'):
        # Use improved cutting method
        print("🔧 Using improved cube cutting for better coverage")
        tree2cubes_improved.cut_point_cloud_improved(point_cloud, cubes_folder, 
                                                     size1=cube_sizes[0], size2=cube_sizes[1], 
                                                     size3=cube_sizes[2], size4=cube_sizes[3],
                                                     min_points=min_points, max_points=max_points, 
                                                     target_points=target_points)
    else:
        # Fall back to original method
        print("⚠️  Using original cube cutting method")
        tree2cubes.cut_point_cloud(point_cloud, cubes_folder, 
                                   size1=cube_sizes[0], size2=cube_sizes[1], 
                                   size3=cube_sizes[2], size4=cube_sizes[3])
    
    txt_files = glob.glob(os.path.join(cubes_folder, "*.txt"))
    
    if debug_analysis:
        print(f"📈 Cube generation analysis:")
        print(f"   Generated {len(txt_files)} cube files")
        
        # Analyze point distribution in cubes
        point_counts = []
        z_positions = []
        
        for txt_file in txt_files:
            data = np.loadtxt(txt_file)
            point_counts.append(data.shape[0])
            z_positions.append(np.mean(data[:, 2]))  # Average Z position of cube
        
        if point_counts:
            print(f"   Points per cube - Min: {min(point_counts)}, Max: {max(point_counts)}, Avg: {np.mean(point_counts):.0f}")
            print(f"   Z coverage - Min: {min(z_positions):.2f}m, Max: {max(z_positions):.2f}m")
            
            # Check for missing coverage
            tree_z_range = np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])
            cube_z_range = max(z_positions) - min(z_positions)
            coverage_ratio = cube_z_range / tree_z_range if tree_z_range > 0 else 0
            print(f"   Vertical coverage: {coverage_ratio:.1%} of tree height")
            
            if coverage_ratio < 0.8:
                print("   ⚠️  WARNING: Low vertical coverage - many tree parts may be missing!")
                print("   💡 Suggestion: Reduce point count thresholds in tree2cubes.py")
    
    # Create flipped versions (data augmentation) if enabled
    if enable_flipping:
        print("🔄 Creating flipped versions for data augmentation...")
        for txt_file in txt_files:
            data = np.loadtxt(txt_file)
            filename = os.path.basename(txt_file)
            
            # Swap x and z coordinates
            flipfile = np.column_stack((data[:, 2], data[:, 1], data[:, 0]))
            flip_path = os.path.join(cubes_folder, filename + "_flip.txt")
            np.savetxt(flip_path, flipfile)
    else:
        print("⏭️  Skipping data augmentation (flipping disabled)")
    
    return len(txt_files)

def run_inference(cubes_folder, run_folder, model_config, model_checkpoint, 
                 save_npy=True, save_ply=False, save_xyz=False, gpu_device="cuda:0"):
    """Run TreePoinTr inference on all cube files"""
    print("🤖 Running TreePoinTr inference...")
    
    inference_output = os.path.join(run_folder, "inference_results")
    
    # Build inference command with configurable parameters
    cmd = [
        "python", "tools/inference.py",
        model_config,
        model_checkpoint,
        "--pc_root", cubes_folder,
        "--out_pc_root", inference_output,
        "--device", gpu_device
    ]
    
    # Add output format flags based on configuration
    if save_npy:
        cmd.append("--save_npy")
    if save_ply:
        cmd.append("--save_ply")
    if save_xyz:
        cmd.append("--save_xyz")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Inference failed:")
        print(result.stderr)
        return False
    
    print("✅ Inference completed successfully!")
    return True

def concatenate_results(run_folder, original_name, separate_flipped=True, combine_all=False):
    """Concatenate all inference results into final tree"""
    print("🔗 Concatenating cube results into complete tree...")
    
    inference_output = os.path.join(run_folder, "inference_results")
    
    # Initialize arrays properly (empty lists, not fixed-size arrays)
    all_points = []
    all_flip_points = []
    
    # Process all inference result directories
    result_dirs = [d for d in os.listdir(inference_output) 
                   if os.path.isdir(os.path.join(inference_output, d))]
    
    processed_count = 0
    
    for result_dir in result_dirs:
        npy_path = os.path.join(inference_output, result_dir, "fine.npy")
        
        if not os.path.exists(npy_path):
            continue
            
        # Load the completed cube
        cube_points = np.load(npy_path)
        
        # Check if this is a flipped version
        if "flip" in result_dir:
            # Swap x and z back for flipped versions
            cube_points_corrected = np.column_stack((cube_points[:, 2], 
                                                   cube_points[:, 1], 
                                                   cube_points[:, 0]))
            all_flip_points.append(cube_points_corrected)
        else:
            all_points.append(cube_points)
        
        processed_count += 1
    
    print(f"📊 Processed {processed_count} cube results")
    
    # Concatenate and save results based on configuration
    saved_files = []
    
    if all_points:
        final_points = np.vstack(all_points)
        print(f"🌳 Main completion: {final_points.shape[0]} points")
        
        # Save main completion
        cloud_main = o3d.geometry.PointCloud()
        cloud_main.points = o3d.utility.Vector3dVector(final_points)
        main_output = os.path.join(run_folder, f"{original_name}_completed.ply")
        o3d.io.write_point_cloud(main_output, cloud_main)
        print(f"✅ Saved main completion: {main_output}")
        saved_files.append(main_output)
    
    # Save flipped version separately if enabled and exists
    if separate_flipped and all_flip_points:
        final_flip_points = np.vstack(all_flip_points)
        print(f"🔄 Flipped completion: {final_flip_points.shape[0]} points")
        
        cloud_flip = o3d.geometry.PointCloud()
        cloud_flip.points = o3d.utility.Vector3dVector(final_flip_points)
        flip_output = os.path.join(run_folder, f"{original_name}_completed_withflips.ply")
        o3d.io.write_point_cloud(flip_output, cloud_flip)
        print(f"✅ Saved flipped completion: {flip_output}")
        saved_files.append(flip_output)
    
    # Save combined version if enabled
    if combine_all and all_points and all_flip_points:
        combined_points = np.vstack([np.vstack(all_points), np.vstack(all_flip_points)])
        print(f"🔗 Combined completion: {combined_points.shape[0]} points")
        
        cloud_combined = o3d.geometry.PointCloud()
        cloud_combined.points = o3d.utility.Vector3dVector(combined_points)
        combined_output = os.path.join(run_folder, f"{original_name}_completed_combined.ply")
        o3d.io.write_point_cloud(combined_output, cloud_combined)
        print(f"✅ Saved combined completion: {combined_output}")
        saved_files.append(combined_output)
    
    return len(saved_files) > 0

def main():
    parser = argparse.ArgumentParser(description="Complete tree processing with TreePoinTr")
    parser.add_argument("input_file", help="Input PLY file path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    if not args.input_file.lower().endswith('.ply'):
        print("❌ Input file must be a PLY file")
        sys.exit(1)
    
    # Extract original name
    original_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    print(f"🌲 Processing tree: {original_name}")
    print(f"📁 Input file: {args.input_file}")
    
    # =================== RESTORATION CONFIGURATION ===================
    # Modify these parameters to customize the tree completion process
    
    # MODEL SELECTION - Choose which trained model to use
    # Available models:
    # - AdaPoinTr_tree_grove_real.pth (recommended for most trees - trained on mixed data)
    # - AdaPoinTr_tree_real_only.pth (best for pure TLS data)
    # - AdaPoinTr_PCN.pth (general objects, not tree-specific)
    # - PoinTr_ShapeNet55.pth (general objects, not tree-specific)
    MODEL_CONFIG = "cfgs/predefhull_models/AdaPoinTr.yaml"
    MODEL_CHECKPOINT = "ckpts/AdaPoinTr_tree_grove_real.pth"
    
    # CUBE CUTTING PARAMETERS - Control how the tree is divided for processing
    # TreePoinTr processes point clouds in chunks. These sizes determine the cube dimensions.
    # Smaller cubes = more detailed completion but longer processing time
    # Larger cubes = faster but may miss fine details
    CUBE_SIZE_1 = 1.0    # First cube size in meters
    CUBE_SIZE_2 = 1.0    # Second cube size in meters  
    CUBE_SIZE_3 = 1.25   # Third cube size in meters
    CUBE_SIZE_4 = 1.8    # Fourth cube size in meters
    
    # DATA AUGMENTATION - Create flipped versions of cubes for better completion
    # Flipping can help with completion quality but doubles processing time
    ENABLE_FLIPPING = True
    
    # OUTPUT FORMATS - Choose what file formats to save
    SAVE_NPY = True      # Save raw numpy arrays (required for concatenation)
    SAVE_PLY = False     # Save individual cube completions as PLY files
    SAVE_XYZ = False     # Save individual cube completions as XYZ files
    
    # PROCESSING PARAMETERS
    # Point cloud normalization - helps model performance
    NORMALIZE_INPUT = True
    
    # Concatenation strategy - how to combine cube results
    SEPARATE_FLIPPED = True  # Save flipped results separately
    COMBINE_ALL = False      # Combine flipped and normal results into one file
    
    # INFERENCE SETTINGS
    GPU_DEVICE = "cuda:0"    # GPU device to use for inference
    BATCH_PROCESSING = True  # Process multiple cubes in batch (faster)
    
    # QUALITY SETTINGS
    TARGET_POINTS_PER_CUBE = 8192  # Target number of points per completed cube
    MIN_POINTS_PER_CUBE = 2730     # Minimum points needed in input cube
    
    # IMPROVED CUBE CUTTING PARAMETERS - Fix coverage issues
    # These parameters control which cubes are saved and processed
    # Lower values = more complete tree coverage, but more processing time
    USE_IMPROVED_CUTTING = True     # Use improved cube cutting (recommended)
    MIN_POINTS_IN_CUBE = 100        # Minimum points to save a cube (was 500-1000 in original)
    MAX_POINTS_IN_CUBE = 8192       # Maximum points per cube (matches model training)
    
    # DOWNSAMPLING TARGET - Why 3000?
    # - TreePoinTr trained on 2048-8192 point clouds
    # - Sweet spot: 2000-4000 points for best quality
    # - Above 4000: diminishing returns + memory issues
    # - Below 2000: insufficient context for completion
    # For powerful GPUs, you can try: 4000-5000 (experimental)
    TARGET_POINTS_DOWNSAMPLE = 3000 # Conservative (reliable)
    # TARGET_POINTS_DOWNSAMPLE = 4000 # Aggressive (powerful GPU)
    
    # ================================================================
    
    print("🔧 Configuration:")
    print(f"   Model: {MODEL_CHECKPOINT}")
    print(f"   Cube sizes: {CUBE_SIZE_1}m, {CUBE_SIZE_2}m, {CUBE_SIZE_3}m, {CUBE_SIZE_4}m")
    print(f"   Data augmentation: {'Enabled' if ENABLE_FLIPPING else 'Disabled'}")
    print(f"   GPU device: {GPU_DEVICE}")
    print(f"   Target points per cube: {TARGET_POINTS_PER_CUBE}")
    print(f"   Improved cutting: {'Enabled' if USE_IMPROVED_CUTTING else 'Disabled'}")
    if USE_IMPROVED_CUTTING:
        print(f"   Min points per cube: {MIN_POINTS_IN_CUBE} (original: 500-1000)")
        print(f"   Max points per cube: {MAX_POINTS_IN_CUBE}")
        print(f"   Downsample target: {TARGET_POINTS_DOWNSAMPLE}")
    
    # Create run folder structure
    run_folder, cubes_folder = create_run_folder()
    print(f"📂 Created run folder: {run_folder}")
    
    try:
        # Load point cloud
        print("📖 Loading point cloud...")
        ply_cloud = o3d.io.read_point_cloud(args.input_file)
        point_cloud = np.asarray(ply_cloud.points)
        print(f"📊 Loaded {point_cloud.shape[0]} points")
        
        # Cut into cubes using configured parameters
        cube_sizes = [CUBE_SIZE_1, CUBE_SIZE_2, CUBE_SIZE_3, CUBE_SIZE_4]
        num_cubes = cut_point_cloud_to_cubes(point_cloud, cubes_folder, cube_sizes, 
                                           ENABLE_FLIPPING, USE_IMPROVED_CUTTING,
                                           MIN_POINTS_IN_CUBE, MAX_POINTS_IN_CUBE, 
                                           TARGET_POINTS_DOWNSAMPLE)
        flip_info = f" (+ flipped versions)" if ENABLE_FLIPPING else ""
        print(f"✂️ Created {num_cubes} cube files{flip_info}")
        
        # Run inference using configured model and parameters
        if not run_inference(cubes_folder, run_folder, MODEL_CONFIG, MODEL_CHECKPOINT,
                           save_npy=SAVE_NPY, save_ply=SAVE_PLY, save_xyz=SAVE_XYZ, 
                           gpu_device=GPU_DEVICE):
            print("❌ Inference failed, stopping...")
            return
        
        # Concatenate results using configured strategy
        if concatenate_results(run_folder, original_name, 
                             separate_flipped=SEPARATE_FLIPPED, combine_all=COMBINE_ALL):
            print(f"🎉 Tree completion successful!")
            print(f"📂 Results saved in: {run_folder}")
        else:
            print("❌ Failed to concatenate results")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 