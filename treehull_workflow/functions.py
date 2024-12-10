# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:10:20 2024

@author: Bornand


functions for for pre-making complete + partial samples
"""

import numpy as np
#from alphashape import alphashape
import trimesh
import random
#from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import os
import laspy
import open3d as o3d

# # Import R's install.packages function
# utils = importr('utils')
# utils.install_packages('alphashape3d')
# utils.install_packages('Morpho')


def downsample_point_cloud(point_cloud, max_points=100000):
    """
    Downsample a point cloud to a maximum number of points.

    Args:
    - point_cloud: PyTorch tensor containing the point cloud data of shape (N, 3).
    - max_points: Maximum number of points to downsample to.

    Returns:
    - downsampled_points: Downsampled point cloud tensor.
    """
    num_points = point_cloud.shape[0]
    if num_points > max_points:
        idx = np.random.choice(num_points, max_points, replace=False)
        downsampled_points = point_cloud[idx]
    else:
        downsampled_points = point_cloud
    return downsampled_points

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def divide_and_sample_point_cloud(point_cloud):
    # Sort points based on z-coordinate
    point_cloud_sorted = point_cloud[np.argsort(point_cloud[:, 2])]

    # Divide into 4 layers
    num_points = len(point_cloud_sorted)
    layer_size = num_points // 4
    layers = np.split(point_cloud_sorted, [layer_size, layer_size*2, layer_size*3])

    # Sample points from each layer
    sampled_points = []
    for i, layer in enumerate(layers):
        if i == 0:
            sample_size = int(layer_size * 0.8)
        elif i == 1:
            sample_size = int(layer_size * 0.2)
        elif i == 2:
            sample_size = int(layer_size * 0.02)
        else:
            sample_size = int(layer_size * 0.001)

        sampled_indices = random.sample(range(len(layer)), sample_size)
        sampled_points.extend(layer[sampled_indices])

    # Convert sampled points to numpy array
    sampled_points = np.array(sampled_points)
    
    return sampled_points


def remove_points_in_sphere(data, percrad = 0.1, centerpos=0.02):

    point_cloud = data
    
    # Ensure the point cloud has the correct shape (N, 3)
    #assert point_cloud.shape[1] == 3, "Point cloud should have shape (N, 3)"
    
    # # Find the point with the highest z-value
    # highest_z_index = np.argmax(point_cloud[:, 2])
    # center = point_cloud[highest_z_index]
    
    #OR: randomply choose a center point out of the 10% of highest points
    
    # Sort points by z-values in descending order
    sorted_indices = np.argsort(point_cloud[:, 2])[::-1]
    sorted_points = point_cloud[sorted_indices]
    
    # Calculate the radius of the sphere
    z_values = sorted_points[:, 2]
    height = np.max(z_values) - np.min(z_values)
    radius = percrad * height
    
    # Select the top 10% (or 2%) of points based on z-values
    # top_percent_count = int(len(point_cloud) * centerpos)
    # top_percent_points = sorted_points[:top_percent_count]
    top_percent_indices = z_values > np.max(z_values) - (height*centerpos)
    
    # Randomly select a center point from the top 10% points
    top_percent_points = sorted_points[top_percent_indices]
    random_index = np.random.choice(top_percent_points.shape[0])
    center = top_percent_points[random_index]


    # Compute the squared radius for distance comparison
    radius_squared = radius ** 2
    
    # Calculate the squared distance of each point from the center
    distances_squared = np.sum((point_cloud - center) ** 2, axis=1)
    
    # Filter out the points within the sphere
    mask = distances_squared > radius_squared
    filtered_point_cloud = point_cloud[mask]
    
    return filtered_point_cloud

    
# Read point cloud from .laz file and convert to numpy array, including the 'Deviation' attribute if it exists
def read_laz_to_numpy(filepath):
    with laspy.open(filepath) as f:
        las = f.read()
        if 'Deviation' in las.point_format.dimension_names:
            points = np.vstack((las.x, las.y, las.z, las.Deviation)).transpose()
        else:    
            points = np.vstack((las.x, las.y, las.z)).transpose()
               
    return points



# Convert NumPy array to R dataframe
def numpy_to_r_dataframe(numpy_array):
    data_frame = robjects.DataFrame({
        'x': robjects.FloatVector(numpy_array[:, 0]),
        'y': robjects.FloatVector(numpy_array[:, 1]),
        'z': robjects.FloatVector(numpy_array[:, 2])
    })
    return data_frame

def points_from_Rashape3d(data, nr_points = 8192, alpha=0.3, file_path= None, method = "vertices"):
    """

    """

    # # Load required R packages
    # robjects.r('''
    #     library(alphashape3d)
    #     library(Morpho)
    # ''')
    
    # Convert numpy array to R dataframe
    r_dataframe = numpy_to_r_dataframe(data)

    r_matrix = robjects.r['as.matrix'](r_dataframe)

    # Compute alphashape object from point cloud
    ashape3d_obj = robjects.r['ashape3d'](r_matrix, alpha=alpha, pert=True) #, eps=0.01

    # Convert alphashape object to mesh3d
    mesh = robjects.r['as.mesh3d'](ashape3d_obj, smooth=False, merge=False)

    # Save alphashapes as ply
    #filename = "D:/PCC_trialset/03334_ashape0.5_RinPy"
    robjects.r['mesh2ply'](mesh, file_path)

    # Randomly sample 8192 points on the surface of the mesh object
    #sampled_points, _ = trimesh.sample.sample_surface(ashape, nr_points)
    
    if method == "samplesurface":
        # Read .ply file
        #mesh = o3d.io.read_triangle_mesh(file_path)
        # Randomly sample 8192 points on the surface of the mesh object
        #point_cloud = mesh.sample_points_uniformly(number_of_points=nr_points)
        ashape = trimesh.load(file_path + ".ply")
        point_cloud, _ = trimesh.sample.sample_surface(ashape, nr_points)
        
    if method == "vertices":
        # Read .ply file
        ashape = trimesh.load(file_path + ".ply")
        # Get vertices of the mesh
        vertices = ashape.vertices
        
        if len(vertices) <= nr_points:
            # sample to nr_points
            fillpoints, _ =  trimesh.sample.sample_surface(ashape, nr_points-len(vertices))
            point_cloud = np.vstack((vertices, fillpoints))
        if len(vertices) > nr_points:
            idx = np.random.choice(len(vertices), size=nr_points, replace=False)
            point_cloud = vertices[idx]

    output = point_cloud
    
    return output  #.contiguous()


def train_test_txt(path_samples, outpath):
    files = os.listdir(path_samples)
    npyfiles = [file for file in files if file.endswith(".npy")]

    test = random.sample(npyfiles, int(len(npyfiles) * 0.2))
    train = np.setdiff1d(npyfiles,test)
    
    file = open(outpath+'test.txt','w')
    for item in test:
     	file.write(os.path.basename(item)+"\n")
    file.close()

    file = open(outpath+'train.txt','w')
    for item in train:
     	file.write(os.path.basename(item)+"\n")
    file.close()


def train_test_txt_predef(path_complete, path_partial, outpath, txt1= "test.txt", txt2 = "train.txt"):
    files_complete = os.listdir(path_complete)
    files_partial = os.listdir(path_partial)
    # list of npy files that are in both folders
    npyfiles = [f for f in files_partial if f in files_complete and f.endswith(".npy")]

    test = random.sample(npyfiles, int(len(npyfiles) * 0.2))
    train = np.setdiff1d(npyfiles,test)
    
    file = open(outpath+txt1,'w')
    for item in test:
     	file.write(os.path.basename(item)+"\n")
    file.close()

    file = open(outpath+txt2,'w')
    for item in train:
     	file.write(os.path.basename(item)+"\n")
    file.close()


def add_suffix_to_files(folder_path, suffix="_local"):
    # Check if the provided folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        old_file_path = os.path.join(folder_path, filename)

        # Check if it's a file (and not a directory)
        if os.path.isfile(old_file_path):
            # Split the file name into name and extension
            file_name, file_extension = os.path.splitext(filename)
            # Create the new file name by adding the suffix
            new_filename = f"{file_name}{suffix}{file_extension}"
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            #print(f"Renamed: {old_file_path} -> {new_file_path}")