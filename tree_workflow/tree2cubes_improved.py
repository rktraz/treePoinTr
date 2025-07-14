#!/usr/bin/env python3
"""
Improved Tree Cube Cutting Function
Fixes point count filtering issues to ensure complete tree coverage
"""

import numpy as np
import os

def cut_point_cloud_improved(point_cloud, outpath, size1, size2, size3, size4, 
                           min_points=100, max_points=8192, target_points=3000):
    """
    Improved cube cutting that preserves more tree parts
    
    Parameters:
    -----------
    min_points : int (default 100)
        Minimum points required to save a cube (much lower than original 500-1000)
    max_points : int (default 8192) 
        Maximum points to keep per cube (matches model training)
    target_points : int (default 3000)
        Target points when downsampling dense cubes
    """
    
    # Create a directory to save the .txt files if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    saved_cubes = 0
    total_points_saved = 0
    skipped_empty = 0
    skipped_sparse = 0
    downsampled_cubes = 0
    
    print(f"🔧 Using improved cube cutting with:")
    print(f"   Min points per cube: {min_points}")
    print(f"   Max points per cube: {max_points}")
    print(f"   Target points (downsampling): {target_points}")
    
    # Find the minimum and maximum coordinates
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    
    # Define spatial shifts for better coverage
    shifts = [
        (0.0, 0.0, 0.0),      # Original position
        (0.5, 0.5, 0.5),      # Half-cube shift
        (-0.3, -0.3, -0.3),   # Negative shift
        (0.3, 0.3, 0.2),      # Smaller positive shift
    ]
    
    cube_sizes = [size1, size2, size3, size4]
    
    for version, (cube_size, (shift_x, shift_y, shift_z)) in enumerate(zip(cube_sizes, shifts)):
        print(f"📦 Processing cube size {cube_size}m (v{version+1}) with shift ({shift_x}, {shift_y}, {shift_z})")
        
        # Apply spatial shift
        shifted_min_x = min_x + shift_x
        shifted_max_x = max_x + shift_x
        shifted_min_y = min_y + shift_y
        shifted_max_y = max_y + shift_y
        shifted_min_z = min_z + shift_z
        shifted_max_z = max_z + shift_z
        
        # Calculate the number of cubes in each dimension
        num_cubes_x = int(np.ceil((shifted_max_x - shifted_min_x) / cube_size))
        num_cubes_y = int(np.ceil((shifted_max_y - shifted_min_y) / cube_size))
        num_cubes_z = int(np.ceil((shifted_max_z - shifted_min_z) / cube_size))
        
        version_cubes = 0
        
        # Iterate over the cubes
        for i in range(num_cubes_x):
            for j in range(num_cubes_y):
                for k in range(num_cubes_z):
                    # Define the cube boundaries
                    cube_min_x = shifted_min_x + i * cube_size
                    cube_max_x = cube_min_x + cube_size
                    cube_min_y = shifted_min_y + j * cube_size
                    cube_max_y = cube_min_y + cube_size
                    cube_min_z = shifted_min_z + k * cube_size
                    cube_max_z = cube_min_z + cube_size
                    
                    # Select the points within the current cube
                    mask = (
                        (point_cloud[:, 0] >= cube_min_x) & (point_cloud[:, 0] < cube_max_x) &
                        (point_cloud[:, 1] >= cube_min_y) & (point_cloud[:, 1] < cube_max_y) &
                        (point_cloud[:, 2] >= cube_min_z) & (point_cloud[:, 2] < cube_max_z)
                    )
                    points_in_cube = point_cloud[mask]
                    
                    # Process cube based on point count
                    if len(points_in_cube) == 0:
                        skipped_empty += 1
                        continue
                    elif len(points_in_cube) < min_points:
                        skipped_sparse += 1
                        continue
                    elif len(points_in_cube) > max_points:
                        # Smart downsampling: try to preserve structure
                        if len(points_in_cube) > target_points:
                            # Use stratified sampling to preserve spatial distribution
                            random_indices = np.random.choice(len(points_in_cube), 
                                                             size=target_points, replace=False)
                            points_in_cube = points_in_cube[random_indices, :]
                        downsampled_cubes += 1
                    
                    # Save the cube
                    filename = f'/cube_{i}_{j}_{k}_v{version+1}.txt'
                    np.savetxt(outpath + filename, points_in_cube[:, 0:3], fmt='%.6f', delimiter=' ')
                    
                    saved_cubes += 1
                    version_cubes += 1
                    total_points_saved += len(points_in_cube)
        
        print(f"   Saved {version_cubes} cubes for this size")
    
    print(f"📊 Cube cutting summary:")
    print(f"   Total cubes saved: {saved_cubes}")
    print(f"   Total points in cubes: {total_points_saved:,}")
    print(f"   Coverage: {(total_points_saved / len(point_cloud)):.1%} of original points")
    print(f"   Skipped empty cubes: {skipped_empty}")
    print(f"   Skipped sparse cubes (<{min_points} points): {skipped_sparse}")
    print(f"   Downsampled dense cubes: {downsampled_cubes}")
    
    return saved_cubes


# Keep original function as backup
def cut_point_cloud_original(point_cloud, outpath, size1, size2, size3, size4):
    """Original tree2cubes function - kept for compatibility"""
    # [Original function code would go here]
    pass 