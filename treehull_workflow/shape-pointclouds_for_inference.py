# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:45:32 2024

@author: Bornand
"""

"""
Make shape-pointclouds from just incomplete tree point clouds

"""
import os
import numpy as np
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


# set input path
inpath = "./testset/"

# set output path
partial_out = "./partial/"

# set output path for alphashape meshes
midpath_partial = "./midashapes/"
if not os.path.exists(midpath_partial):
    os.makedirs(midpath_partial)




# List all LAS/LAZ files in the input folder that are not yet in output folder
out_files = set(os.listdir(partial_out))
npy_files = [f for f in os.listdir(inpath) if f.endswith(('.laz', '.las')) and f not in out_files]


alpha = 0.3
npoints = 8192

for file_name in npy_files:

    print(file_name)
    
    file = functions.read_laz_to_numpy(os.path.join(inpath, file_name))    
    
    pc = file[:,0:3]

    
    pc_small = functions.downsample_point_cloud(pc, max_points=100000)
    
    # normalize it (otherwise alphashape has problems)
    centroid = np.mean(pc_small, axis=0)
    pc_normed = pc_small - centroid
    m = np.max(np.sqrt(np.sum(pc_normed**2, axis=1)))
    pc_normed = pc_normed / m
    

    
    try:
        
        partial = functions.points_from_Rashape3d(pc_normed, nr_points = npoints/2, alpha=alpha, file_path= os.path.join(midpath_partial,os.path.basename(file_name))) 
        
        # denormalize it to adapt for the original input
        partial = partial * m
        partial = partial + centroid

        # Save the resulting point cloud to a new .npy file
        partial_filename = os.path.basename(file_name)
        np.save(os.path.join(partial_out, partial_filename), partial)
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()



