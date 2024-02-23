# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:25:14 2023

"""
import numpy as np
import pandas as pd
import os
import glob
import json
import random
import shutil


def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points
    from: https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy    
    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter


def read_data_table(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) < 2:
            raise ValueError("File must have at least two lines (header and data).")

        header = lines[0].strip().split(' ')
        data = [line.strip().split(' ') for line in lines[1:]]
        data = np.array(data).astype(float)
        
        return data, header


def split_points_by_sourceid(data):
    split_data = {}
    
        # if PointSourceID does not exists: rename GpsTime as PointSourceId
        # get unique values of PointSourceId
            # if max(values) > 20:
                # bin them into 10 classes
                
    if 'PointSourceId' not in data.columns:
        data['PointSourceId'] = data['GpsTime']
                
    if max(data['PointSourceId']) > 20:            
        # Calculate class boundaries
        num_classes = 10
        class_boundaries = pd.qcut(data['PointSourceId'], q=num_classes, labels=False, duplicates='drop')
        # Add the new column to the DataFrame
        data['SourceId_Class'] = class_boundaries
    else:
       data['SourceId_Class'] = data['PointSourceId']

    for source_id, group in data.groupby("SourceId_Class"):
        split_data[source_id] = group
    #print("split by source id")    
    return split_data





def mksamples_real(fulltree_path, outpath, start_count=0, stop_count=100, boxsize=0.5):
    
    # make data output directory 
    if not os.path.exists(outpath+"/train/"):
        os.makedirs(outpath+"/train/")
    outpath = outpath+"/train/"
    
    n_points = 8192

    # loop through complete trees
    for item in glob.glob(fulltree_path +"*.xyz"):
    
        # items = glob.glob(full_tree_path +"*.txt")  
        # print(int(sys.argv[1]))
        # item = items[int(sys.argv[1])]
            
        # get the tree ID (banr) from string
        # treename = re.findall(r'\d+', item)
        # banr = list(map(int, treename))
        # treename = treename[0]
        treename = os.path.basename(item)
        treename = os.path.splitext(treename)[0]
        #print(item)
        
        pc, pc_header = read_data_table(item)
        
        # center point cloud if coordinates are too big
        if abs(min(pc[:, 0])) > 100:
           pc[:, 0] = pc[:, 0] + (0-min(pc[:, 0]))
           pc[:, 1] = pc[:, 1] + (0-min(pc[:, 1]))
        if abs(min(pc[:, 2])) > 50:   
           pc[:, 2] = pc[:, 2] + (0-min(pc[:, 2]))
        
        
        
        # generate point cloud samples of bb size s for random locations in tree
        count = start_count
        # number of samples:
        n = stop_count
        while count in range(n):
            # select a random point (row)
            number_of_rows = pc.shape[0]
            random_indices = np.random.choice(number_of_rows, size=1, replace=False)
            # display random rows
            #print("\nRandom row:")
            row = pc[random_indices, :]
            #print(row)
            
            
            # bounding box selection around point
            # box size?
            s = 0.5
            
        
            min_x = float(row[0,0])-s
            max_x = float(row[0,0])+s
            min_y = float(row[0,1])-s
            max_y = float(row[0,1])+s
            min_z = float(row[0,2])-s
            max_z = float(row[0,2])+s
            
            pc_bb_filter = bounding_box(pc, min_x, max_x, min_y, max_y, min_z, max_z)
            pc_sample = pc[pc_bb_filter,:]
        
            
            if len(pc_sample)>n_points:
            
                # # translate sample point cloud back into original
                # sample[:, 0] = sample[:, 0]-100
                # sample[:, 1] = sample[:, 1]-100
            
                # downsample complete point cloud to n_points points
                number_of_rows = pc_sample.shape[0]
                random_indices = np.random.choice(number_of_rows, size=n_points, replace=False)
                pc_sample = pc_sample[random_indices,:]
                
                # make data output directory for each tree
                if not os.path.exists(outpath+"complete/"+treename):
                   os.makedirs(outpath+"complete/"+treename)
                                 
        
        
                pc_sample_pd = pd.DataFrame(data=pc_sample, columns=pc_header)
                # split ppc_sample by PointSourceId
                arrays = split_points_by_sourceid(pc_sample_pd)
                
        
                # Generate combinations of arrays
                combinations = set()
                count_partial = 0
                #while count_partial in range(8):
                for file1, array1 in arrays.items():
                    for file2, array2 in arrays.items():
                        if file1 != file2:
                            combination_key = frozenset([file1, file2])
                            if combination_key not in combinations and len(array1)>10 and len(array2)>10: # no combination whre only one array contains points
                                combined = np.concatenate((array1, array2), axis=0)
                                combinations.add(combination_key)
                                #print("combined scans")
                                # only save partial cloud if enough points in partial cloud sample (more than 1/3 of complete cloud)
                                if n_points/3 < len(combined):
                                    
                                    # make data output directory for each tree
                                    outdir = outpath+"partial/"+treename+'/'+treename+'_'+str(count)+'_size'+str(s*2)
                                    if not os.path.exists(outdir):
                                       os.makedirs(outdir)
        
                                    # Save as .npy file
                                    if count_partial < 9:
                                        npy_file = treename+'_'+str(count)+'_size'+str(s*2)+'_'+str(file1)+"_"+str(file2)+".npy"
                                        np.save(outdir+"/"+npy_file, combined) # [:,0:3]
                                        count_partial += 1
                                    #print(f"Saved {npy_file}")
        
    
        
                
                if os.path.exists(outpath+"partial/"+treename+'/'+treename+'_'+str(count)+'_size'+str(s*2)):
                             
                    # save complete point cloud sample
                    outname = outpath+"complete/"+treename+'/'+treename+'_'+str(count)+'_size'+str(s*2)
                    np.save(outname, pc_sample) # pc_sample[:,0:3]
                    
                    # only count up if sample gets saved
                    count += 1





def traintest_json(complete_dir_train, outpath, dataset_name):

    dict_lst = [] 
       
    banrs = os.listdir(complete_dir_train) # taxonomy id
    for banr in banrs: 
        filenames = [os.path.splitext(f)[0] for f in os.listdir(complete_dir_train+"/"+banr)]
        #filenames_test = [os.path.splitext(f)[0] for f in os.listdir(complete_dir_test+"/"+banr)]
        test = random.sample(filenames, int(len(filenames) * 0.2))
        train = np.setdiff1d(filenames,test).tolist()
        
        dictionary = {
            "taxonomy_id": str(banr),
            "taxonomy_name": str(banr),
            "test": test,
            "train": train,
            "val": []
            }
        
        dict_lst.append(dictionary)
        
        # move the test files to the test folder structure
        for file in test:
     
            # Create the full paths for source and destination
            source_path = complete_dir_train + "/"+ banr + "/" + file +".npy"
            destination_directory_complete = outpath + "test/complete/" + banr + "/"
            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_directory_complete):
                os.makedirs(destination_directory_complete)
            # Construct the destination path
            destination_path = destination_directory_complete + file +".npy"
            # Move the file to the destination
            shutil.move(source_path, destination_path)
        
            # Create the full paths for source and destination
            source_path = outpath + "train/partial/"+ banr + "/" + file
            destination_directory_partial = outpath + "test/partial/" + banr +"/"
            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_directory_partial):
                os.makedirs(destination_directory_partial)
            # Construct the destination path
            destination_path = destination_directory_partial + file 
            # Move the file to the destination
            shutil.move(source_path, destination_path)
        

    json_object = json.dumps(dict_lst, indent=4)
    with open(outpath+dataset_name+".json", "w") as outfile:
        outfile.write(json_object)



