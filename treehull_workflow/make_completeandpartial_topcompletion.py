# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:45:32 2024

@author: Bornand
"""

"""
main for pre-making complete + partial samples from full single tree point clouds
"""
import os
import numpy as np
import random
import functions
# Import R's install.packages function
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
utils = importr('utils')
# install R packages
# utils.install_packages('alphashape3d')
# utils.install_packages('Morpho')

# path to R installation
os.environ['R_HOME'] = '~/R/x86_64-pc-linux-gnu-library/4.1'

# Load required R packages
robjects.r('''
    library(alphashape3d)
    library(Morpho)
''')


# path to full tree point cloud files
inpath = "./data/simplehull/simplehull_pc"

midpath_partial = "./ashapes_partial"
if not os.path.exists(midpath_partial):
    os.makedirs(midpath_partial)
midpath_complete = "./ashapes_complete"
if not os.path.exists(midpath_complete):
    os.makedirs(midpath_complete)

gt_out = "./data/predefhull/complete"
if not os.path.exists(gt_out):
    os.makedirs(gt_out)
partial_out = "./data/predefhull/partial"
if not os.path.exists(partial_out):
    os.makedirs(partial_out)


# List all npy files in the input folder
npy_files = [f for f in os.listdir(inpath) if f.endswith('.npy')]
# List all  files in the input folder that are not yet in output folder
#gt_out_files = set(os.listdir(gt_out))
partial_out_files = set(os.listdir(partial_out))
npy_files = [f for f in os.listdir(inpath) if f.endswith('.npy') and f not in partial_out_files]

#
npoints = 8192

for file_name in npy_files:

    print(file_name)
    out_filename = os.path.basename(file_name)
    file= np.load(os.path.join(inpath, file_name))
    pc = file[:,0:3]

    max_points = random.randint(75000, 110000)
    pc_small = functions.downsample_point_cloud(pc, max_points=max_points)
    pc_normed = functions.pc_norm(pc_small) 
    
    # set alpha value for shape
    #alpha = 0.3
    alpha = round(random.uniform(0.25, 0.34), 2)
    # set radius for sphere (percentage of tree height)
    percrad = round(random.uniform(0.02, 0.5), 2)
    # set percentage of height layer from which sphere centre is chosen
    centerpos = round(random.uniform(0.02, 0.25), 2)
    
    try:
        # compute alphashape and generated points from vertices and surface
        gt = functions.points_from_Rashape3d(pc_normed, nr_points = npoints, alpha=alpha, file_path= os.path.join(midpath_complete, out_filename))
        # Save the resulting point cloud to a new .npy file
        np.save(os.path.join(gt_out, out_filename), gt)
        
        # remove upper points with sphere method
        partial_input = functions.remove_points_in_sphere(pc_normed, percrad = percrad, centerpos= centerpos)
        # compute alphashape and generated points from vertices and surface
        partial = functions.points_from_Rashape3d(partial_input, nr_points = random.randrange(int(npoints * 1/4) , int(npoints * 3/4)), alpha=alpha, file_path= os.path.join(midpath_partial, out_filename)) 
        
        # Save the resulting point cloud to a new .npy file
        #partial_filename = "r1_"+os.path.basename(file_name)
        np.save(os.path.join(partial_out, out_filename), partial)
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")    

#---------------------------------------------------------------------------------------




