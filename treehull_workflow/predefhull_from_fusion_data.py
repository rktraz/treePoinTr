# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:01:37 2024

@author: Bornand
"""

# make complete and partial point clouds via alphashapes, if co-registered data from different sensors already exists (real partials)

import os
import numpy as np
import laspy
import open3d as o3d
import random

import functions

# Import R's install.packages function
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
utils = importr('utils')
# utils.install_packages('alphashape3d')
# utils.install_packages('Morpho')

#os.environ['R_HOME'] = '~/R/x86_64-pc-linux-gnu-library/4.1'
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.3'

# Load required R packages
robjects.r('''
    library(alphashape3d)
    library(Morpho)
''')

# Function to normalize point cloud
# def pc_norm(pc):
#     """ pc: NxC, return NxC """
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc

# def pc_norm(pc):
#     """ pc: NxC, return NxC with only the first three columns normalized """
#     pc_first_three = pc[:, :3]
#     pc_rest = pc[:, 3:]
    
#     centroid = np.mean(pc_first_three, axis=0)
#     pc_first_three = pc_first_three - centroid
#     m = np.max(np.sqrt(np.sum(pc_first_three**2, axis=1)))
#     pc_first_three = pc_first_three / m
    
#     pc_normalized = np.hstack((pc_first_three, pc_rest))
#     return pc_normalized


# Read point cloud from .laz file and convert to numpy array
def read_laz_to_numpy(filepath):
    with laspy.open(filepath) as f:
        las = f.read()
        points = np.vstack((las.x, las.y, las.z, las.platform)).transpose()            
    return points



# Save point clouds as .xyz files
def save_to_xyz(pc, filename):
    np.savetxt(filename, pc, fmt="%.6f", delimiter=' ')

###################
# define paths
# pc1_path = "F:/PCC_singleTrees/UTM_fusion_ALShd_MLS/test/" #singletrees_clean
# #pc2_path = "F:/PCC_singleTrees/from_pytreedb/ALS"
# outpath1 = "F:/PCC_singleTrees/UTM_fusion_ALShd_MLS/singletrees_ALS+MLS_ashapes/"
# outpath2 = "F:/PCC_singleTrees/UTM_fusion_ALShd_MLS/singletrees_MLS_ashapes/"
# midpath = "F:/PCC_singleTrees/UTM_fusion_ALShd_MLS/singletrees_mid_plys/"
pc1_path = "./singletrees_clean_sub004/"
gt_out = "./singletrees_ALS+MLS_ashapes_v2/"
partial_out = "./singletrees_MLS_ashapes_v2/"
midpath_complete = "./singletrees_mid_fullplys/"
midpath_partial =  "./singletrees_mid_partialplys/"
partial_out_xyz = "./singletrees_MLS_ashapes_v2_xyz/"

if not os.path.exists(gt_out):
    os.makedirs(gt_out)

if not os.path.exists(partial_out):
    os.makedirs(partial_out)    
    
if not os.path.exists(midpath_complete):
    os.makedirs(midpath_complete)     
    
if not os.path.exists(midpath_partial):
    os.makedirs(midpath_partial)       

if not os.path.exists(partial_out_xyz):
    os.makedirs(partial_out_xyz) 

# List all LAS/LAZ files in the output folder
gt_out_files = set(os.listdir(partial_out))
#laz_files = [f for f in os.listdir(pc1_path) if f.endswith(('.laz', '.las')) and f not in gt_out_files]
laz_files = [
    f for f in os.listdir(pc1_path)
    if (f.endswith('.las') or f.endswith('.laz'))
    and not any(f.startswith(gt_file[:13]) for gt_file in gt_out_files)
]


for file_name in laz_files:
    
    # Read the first .laz file
    gt_file = file_name
    print(gt_file)
    

        
    cloud = read_laz_to_numpy(os.path.join(pc1_path, gt_file))

    
    filename = os.path.splitext(os.path.basename(file_name))[0]
    #filename = filename[:13]
    alpha = 0.3   # or change this to a different alpha value
    npoints = 8192   # number of points per sample
    

    ### full pc
    #pc= xyz[:, :3]
    pc = cloud[:, :3]
    pc_small = functions.downsample_point_cloud(pc, max_points=75000)  # alphashape computation works better if pc is not too big
    
    # normalize full pc
    centroid = np.mean(pc_small, axis=0)
    pc_normed = pc_small - centroid
    m = np.max(np.sqrt(np.sum(pc_normed**2, axis=1)))
    pc_normed = pc_normed / m
    #pc_normed = np.round(pc_normed, 3)
    
    
    # separate by platform
    mls = cloud[cloud[:, 3] == 0] # for data where platform info is stored in column 4, here MLS = 0
    pc2_small = functions.downsample_point_cloud(mls[:, :3], max_points=75000)

    # normalize partial pc
    centroid2 = np.mean(pc2_small, axis=0)
    pc_normed2 = pc2_small - centroid2
    m2 = np.max(np.sqrt(np.sum(pc_normed2**2, axis=1)))
    pc_normed2 = pc_normed2 / m2
    
    # Convert the NumPy array to an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_normed2[:, :3])

    # # Apply Statistical oulier removal
    inlier, idx = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    
    # Convert the downsampled point cloud back to a NumPy array
    pc_normed2_f = np.asarray(inlier.points)
    #save_to_xyz(pc_normed2_f, os.path.join(partial_out_xyz, filename[:13]+"_SORfiltered.xyz"))
    
        
    try:
        
        out_filename1 = filename[:13]+"_gt"
        gt = functions.points_from_Rashape3d(pc_normed, nr_points = npoints, alpha=alpha, file_path= os.path.join(midpath_complete, out_filename1))
        # denormalize
        gt = gt * m
        gt = gt + centroid
        # Save the resulting point cloud to a new .npy file
        np.save(os.path.join(gt_out, out_filename1), gt)
        
        out_filename2 = filename[:13]+"_mls"
        #partial_input = functions.remove_points_in_sphere(pc_normed, percrad = percrad, centerpos= centerpos)
        partial = functions.points_from_Rashape3d(pc_normed2_f, nr_points = random.randrange(int(npoints * 1/4) , int(npoints * 3/4)), alpha=alpha, file_path= os.path.join(midpath_partial, out_filename2)) 
        # denormalize
        partial = partial * m2
        partial = partial + centroid2
        # Save the resulting point cloud to a new .npy file
        np.save(os.path.join(partial_out, out_filename2), partial)
        
        save_to_xyz(partial, os.path.join(partial_out_xyz, out_filename2+".xyz"))
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")   


print("done")
