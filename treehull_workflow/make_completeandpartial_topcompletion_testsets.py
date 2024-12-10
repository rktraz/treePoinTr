# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:45:32 2024

@author: Bornand
"""

"""
pre-making full + partial samples for evaluation
remove predefined fractions of height from tree top
"""
import os
import numpy as np
import random
import glob
import laspy
import functions
# Import R's install.packages function
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
utils = importr('utils')
# install R packages
# utils.install_packages('alphashape3d')
# utils.install_packages('Morpho')

# path to R installation
#os.environ['R_HOME'] = '~/R/x86_64-pc-linux-gnu-library/4.1'
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.3'

# Load required R packages
robjects.r('''
    library(alphashape3d)
    library(Morpho)
''')


# Function to filter point cloud based on 'Deviation' value
def filter_by_deviation(pc, deviation_index):
    deviation_values = pc[:, deviation_index]
    median_deviation = np.median(deviation_values)
    std_deviation = np.std(deviation_values)
    threshold = median_deviation + 2 * std_deviation
    
    filtered_pc = pc[deviation_values <= threshold]
    return filtered_pc



pc1_path = "F:/PCC_singleTrees/from_pytreedb/TLS/"
pc2_path = "F:/PCC_singleTrees/from_pytreedb/ALS"


midpath_partial = "F:/PCC_singleTrees/ashapes_partial"
if not os.path.exists(midpath_partial):
    os.makedirs(midpath_partial)
midpath_complete = "F:/PCC_singleTrees/ashapes_complete"
if not os.path.exists(midpath_complete):
    os.makedirs(midpath_complete)

gt_out = "F:/PCC_singleTrees/from_pytreedb/TLS+ALS_ashapes_forvalidation/"
if not os.path.exists(gt_out):
    os.makedirs(gt_out)
 
# predefined fraction of tree height to remove from top    
toremove = [0.1, 0.2, 0.3, 0.4, 0.5]    

for nrem in toremove:
    
    partial_out = "F:/PCC_singleTrees/from_pytreedb/TLS+ALS_ashapes_partial_"+str(nrem)+"/"
    if not os.path.exists(partial_out):
        os.makedirs(partial_out)
    
    
    # List all npy files in the input folder
    laz_files = [f for f in os.listdir(pc1_path) if f.endswith(('.laz', '.las')) ]
    
    
    #alpha = 0.3
    npoints = 8192
    alpha = 0.3
    #percrad = 0.05  # -> change for different test sets
    percrad = nrem
    centerpos = 0.02
    
    
    for file_name in laz_files:
    
        # Step 1: Read the first .laz file
        filename = os.path.splitext(os.path.basename(file_name))[0]
        print(filename)
        gt_file = file_name
        pc1 = functions.read_laz_to_numpy(os.path.join(pc1_path, gt_file))
        
        # Step 2: Find and read the second .laz file
        prefix = gt_file[:14]
        folder_path = pc2_path  # Update this to the correct folder path
        search_pattern = os.path.join(folder_path, prefix + "*.laz")
        files = glob.glob(search_pattern)
        
        # if not files:
        #     raise FileNotFoundError("No matching .laz file found in the specified folder.")
        if files:
        
            file2 = files[0]
            pc2 = functions.read_laz_to_numpy(file2)
            
            # Step 4: Filter the point clouds based on 'Deviation' if it exists
            if 'Deviation' in laspy.open(os.path.join(pc1_path, gt_file)).read().point_format.dimension_names:
                print("deviation filter on gt")
                deviation_index = 3  # Deviation is the 4th column (0-indexed)
                pc1 = filter_by_deviation(pc1, deviation_index)
                
            if 'Deviation' in laspy.open(file2).read().point_format.dimension_names:
                print("deviation filter on partial")
                deviation_index = 3  # Deviation is the 4th column (0-indexed)   
                pc2 = filter_by_deviation(pc2, deviation_index)
            
            # Step 3: Merge the arrays
            pc1 = pc1[:,:3]
            pc2 = pc2[:,:3]
            merged_pc = np.vstack((pc1, pc2))
            
            
            pc_small = functions.downsample_point_cloud(merged_pc, max_points=100000)
            #pc_normed = functions.pc_norm(pc_small) 
            
            # normalize it to fit the model on ShapeNet-55/34
            centroid = np.mean(pc_small, axis=0)
            pc_normed = pc_small - centroid
            m = np.max(np.sqrt(np.sum(pc_normed**2, axis=1)))
            pc_normed = pc_normed / m
            
    
        
            try:
                out_filename = filename+"_gt"
                # compute alphashape and generated points from vertices and surface
                gt = functions.points_from_Rashape3d(pc_normed, nr_points = npoints, alpha=alpha, file_path= os.path.join(midpath_complete, out_filename))
                # denormalize
                gt = gt * m
                gt = gt + centroid
                # Save the resulting point cloud to a new .npy file
                np.save(os.path.join(gt_out, out_filename), gt)
                
                out_filename = filename+"_partial"
                # remove upper points with sphere method
                partial_input = functions.remove_points_in_sphere(pc_normed, percrad = percrad, centerpos= centerpos)
                # compute alphashape and generated points from vertices and surface
                partial = functions.points_from_Rashape3d(partial_input, nr_points = random.randrange(int(npoints * 1/4) , int(npoints * 3/4)), alpha=alpha, file_path= os.path.join(midpath_partial, out_filename)) 
                # denormalize
                partial = partial * m
                partial = partial + centroid
                # Save the resulting point cloud to a new .npy file
                np.save(os.path.join(partial_out, out_filename), partial)
            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")    






