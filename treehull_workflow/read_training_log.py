# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:21:12 2024

@author: Bornand
"""
# read training log files and make graphs showing training progress (Chamfer distance vs. epoch)

import re
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

#modelnames = ['AdaPoinTr_finetune_200', 'AdaPoinTr1', 'AdaPoinTr1_300', 'AdaPoinTr2', 'AdaPoinTr3', 'AdaPoinTr4', 'AdaPoinTr5', 'AdaPoinTr6', 'AdaPoinTr7', 'AdaPoinTr8', 'AdaPoinTr9']
#modelnames = ['AdaPoinTr_finetune_200', 'AdaPoinTr1_300', 'AdaPoinTr2', 'AdaPoinTr3', 'AdaPoinTr4', 'AdaPoinTr5', 'AdaPoinTr6',  'AdaPoinTr7', 'AdaPoinTr8',  'AdaPoinTr9', 'AdaPoinTr10', 'AdaPoinTr11']
modelnames = ['AdaPoinTr2', 'AdaPoinTr3', 'AdaPoinTr4', 'AdaPoinTr5', 'AdaPoinTr6',  'AdaPoinTr7', 'AdaPoinTr8', 'AdaPoinTr10', 'AdaPoinTr11', 'AdaPoinTr12']

path_template = "./experiments/{}/predefhull_models/*/*.log"

outpath = "./experiments/vis/"
if not os.path.exists(outpath):
    os.makedirs(outpath)

for model in modelnames:

    #path = "D:/treePoinTr_experiments/fromscratch_predefhull_AdaPointr_main1/"
    
    # Construct the path for the current model
    log_path_pattern = path_template.format(model)
    
    # Use glob to find all .log files matching the pattern
    logfiles = glob.glob(log_path_pattern)
    
    #logfile = "D:/treePoinTr_experiments/fromscratch_predefhull_AdaPointr_main1/20240806_113806.log"
    #logfile = os.path.join(path,"20240806_113806.log")
    
    # Initialize an empty list to store extracted data
    data = []
    
    # Define a regular expression pattern to match the lines of interest
    
    pattern = re.compile(rf"(\d{{4}}-\d{{2}}-\d{{2}}) (\d{{2}}:\d{{2}}:\d{{2}},\d{{3}}) - {model} - INFO - \[Validation\] EPOCH: (\d+)\s+Metrics = \['(\d+\.\d+)',\s*'(\d+\.\d+)',\s*'(\d+\.\d+)',\s*'(\d+\.\d+)'\]")
    
    
    # 2024-08-13 16:16:21,258 - AdaPoinTr1 - INFO - [Validation] EPOCH: 145  Metrics = ['0.4736', '15.4742', '1.8576', '0.0000']
    
    # Open the log file and read line by line
    with open(logfiles[0], 'r') as file: 
        for line in file:
            match = re.search(pattern, line)
            if match:
                date = match.group(1)
                time = match.group(2)
                epoch = int(match.group(3))
                metrics = [float(match.group(4)), float(match.group(5)), float(match.group(6)), float(match.group(7))]
                data.append([date, time, epoch] + metrics)
                print(f"match: {line.strip()}")
            # else:
            #     print(f"No match: {line.strip()}")
    
    
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=['Date', 'Time', 'Epoch', 'F-Score', 'CDL1', 'CDL2', 'EMDistance'])
    
    # Display the DataFrame
    #print(df)
    
    # Save the DataFrame to a CSV file (optional)
    df.to_csv(os.path.join(outpath, model+".csv"), index=False)
    
    #-----------------------------------------------------------
    
    
    
    df_col = 'CDL2'
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for Metric3
    Q1 = df[df_col].quantile(0.25)
    Q3 = df[df_col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for non-outlier values
    #lower_bound = Q1 - 1.5 * IQR
    lower_bound = 0
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    filtered_df = df[(df[df_col] >= lower_bound) & (df[df_col] <= upper_bound)]
    
    # Plotting the filtered data
    plt.figure(figsize=(20, 12))
    plt.plot(filtered_df['Epoch'], filtered_df[df_col], marker='o', linestyle='-', color='b')
    
    # unfiltered
    #plt.plot(df['Epoch'], df[df_col], marker='o', linestyle='-', color='r')
    
    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel(df_col)
    plt.title(df_col+ ' vs. Epoch')
    #plt.title(df_col+ '  vs. Epoch (After Removing Outliers)')
    
    # Optional: Adding grid for better readability
    plt.grid(True)
    
    # Save the plot as a .png file
    plt.savefig(os.path.join(outpath, model+"_"+df_col+ '_vs_Epoch.png'))
    
    # Save the plot as a .svg file
    plt.savefig(os.path.join(outpath, model+"_"+df_col+ '_vs_Epoch.svg'))
    
    # Show the plot
    plt.show()


#----------------------------------------------

# Define the folder containing the CSV files
folder_path = outpath

# Initialize a plot
plt.figure(figsize=(20, 12))

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        df_col = 'CDL2'
        
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) for Metric3
        Q1 = df[df_col].quantile(0.25)
        Q3 = df[df_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for non-outlier values
        #lower_bound = Q1 - 1.5 * IQR
        lower_bound = 0
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        filtered_df = df[(df[df_col] >= lower_bound) & (df[df_col] <= upper_bound)]
        
        # Plotting the filtered data
        plt.plot(filtered_df['Epoch'], filtered_df[df_col], label=filename)
        
        # Plot CDL2 vs. Epoch
        #plt.plot(df['Epoch'], df['CDL2'], label=filename)
        
# Add labels and a title
plt.xlabel('Epoch', fontsize=30)
plt.ylabel(df_col, fontsize=30)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.title(df_col +' vs. Epoch for Different Models (After Removing Outliers)', fontsize=30)
plt.legend(title='model name', loc='best', fontsize=20)
# Restrict the y-axis to values between 0 and 1.5
plt.ylim(0, 1.5)
plt.grid(True)

# Save the plot as a .png file
plt.savefig(os.path.join(folder_path, "allmodels_"+df_col+ '_vs_Epoch_filtered.png'))

# Save the plot as a .svg file
plt.savefig(os.path.join(folder_path, "allmodels_"+df_col+ '_vs_Epoch_filtered.svg'))

# Show the plot
plt.show()

#--------------------------------------------------

# Initialize a plot
plt.figure(figsize=(20, 12))

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        df_col = 'CDL2'
        
        
        # Plotting the filtered data
        plt.plot(df['Epoch'], df[df_col], label=filename)
        

# Add labels and a title
plt.xlabel('Epoch', fontsize=30)
plt.ylabel(df_col, fontsize=30)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title(df_col +' vs. Epoch for Different Models', fontsize=30)
plt.legend(title='Filename', loc='best', fontsize=20)
# Restrict the y-axis to values between 0 and 1.5
plt.ylim(0, 2)
plt.grid(True)

# Save the plot as a .png file
plt.savefig(os.path.join(folder_path, "allmodels_"+df_col+ '_vs_Epoch_nofilter.png'))

# Save the plot as a .svg file
plt.savefig(os.path.join(folder_path, "allmodels_"+df_col+ '_vs_Epoch_nofilter.svg'))

# Show the plot
plt.show()


# Initialize a plot
plt.figure(figsize=(20, 12))

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        df_col = 'CDL1'
        
        
        # Plotting the filtered data
        plt.plot(df['Epoch'], df[df_col], label=filename)
        

# Add labels and a title
plt.xlabel('Epoch', fontsize=30)
plt.ylabel(df_col, fontsize=30)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title(df_col +' vs. Epoch for Different Models', fontsize=30)
plt.legend(title='Filename', loc='best', fontsize=20)
# Restrict the y-axis to values between 0 and 1.5
plt.ylim(0, 2)
plt.grid(True)

# Save the plot as a .png file
plt.savefig(os.path.join(folder_path, "allmodels_"+df_col+ '_vs_Epoch_nofilter.png'))

# Save the plot as a .svg file
plt.savefig(os.path.join(folder_path, "allmodels_"+df_col+ '_vs_Epoch_nofilter.svg'))

# Show the plot
plt.show()
