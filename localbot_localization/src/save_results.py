#!/usr/bin/env python3

# 3rd-party

import os
from visualization_msgs.msg import *
from localbot_core.src.utilities import *
from colorama import Fore
from datetime import datetime
import yaml
import pandas as pd
import shutil
import matplotlib.pyplot as plt

class SaveResults():
    """
    class to save results
    """
    
    def __init__(self, output, model_path, seq_path, overwrite):
        
        # attribute initializer
        self.output_folder = f'{os.environ["HOME"]}/results/localbot/{output}'
        self.model_path = model_path
        self.seq_path = seq_path
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)  # Create the new folder
        elif overwrite:
            print(f'Overwriting folder {self.output_folder}')
            shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder)  # Create the new folder
        else:
            print(f'{Fore.RED} {self.output_folder} already exists... Aborting SaveResults initialization! {Fore.RESET}')
            exit(0)
        
        
        
        dt_now = datetime.now() # current date and time
        config = {'user'       : os.environ["USER"],
                  'date'       : dt_now.strftime("%d/%m/%Y, %H:%M:%S"),
                  'model_path' : self.model_path,
                  'seq_path'   : self.seq_path}
        
        with open(f'{self.output_folder}/config.yaml', 'w') as file:
            yaml.dump(config, file)
        
        self.frame_idx = 0 # make sure to save as 00000
        self.csv = pd.DataFrame(columns=('frame', 'position_error (m)', 'rotation_error (rads)'))
        
        print('SaveResults initialized properly')

    def saveTXT(self, real_transformation, predicted_transformation):
        
        filename = f'frame-{self.frame_idx:05d}'
        
        write_transformation(f'{self.output_folder}/{filename}.real.pose.txt', real_transformation)
        write_transformation(f'{self.output_folder}/{filename}.predicted.pose.txt', predicted_transformation)
  

    def updateCSV(self, position_error, rotation_error):
        row = {'frame' : f'{self.frame_idx:05d}', 
                           'position_error (m)' : position_error,
                           'rotation_error (rads)' : rotation_error}
        
        self.csv = self.csv.append(row, ignore_index=True)  
        
    def saveCSV(self):
        # save averages values in the last row
        mean_row = {'frame'                 : 'mean_values', 
                    'position_error (m)'    : self.csv.mean(axis=0).loc["position_error (m)"],
                    'rotation_error (rads)' : self.csv.mean(axis=0).loc["rotation_error (rads)"]}
        
        
        median_row = {'frame'                 : 'median_values', 
                      'position_error (m)'    : self.csv.median(axis=0).loc["position_error (m)"],
                      'rotation_error (rads)' : self.csv.median(axis=0).loc["rotation_error (rads)"]}
        
        self.csv = self.csv.append(mean_row, ignore_index=True)  
        self.csv = self.csv.append(median_row, ignore_index=True) 
        
        
        print(self.csv)
        self.csv.to_csv(f'{self.output_folder}/errors.csv', index=False, float_format='%.5f')

    def saveErrorsFig(self):
        frames_array = self.csv.iloc[:-2]['frame'].to_numpy().astype(int)
        
        pos_error_array = self.csv.iloc[:-2]['position_error (m)'].to_numpy()
        rot_error_array = self.csv.iloc[:-2]['rotation_error (rads)'].to_numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle('position and rotation errors')
        ax1.plot(frames_array, pos_error_array, 'cyan',  label='position error')
        ax2.plot(frames_array, rot_error_array, 'navy', label='rotation error')
        ax2.set_xlabel('frame idx')
        ax2.set_ylabel('[rads]')
        ax1.set_ylabel('[m]')
        ax1.legend()
        ax2.legend()
        plt.savefig(f'{self.output_folder}/errors.png')
        
        
    def step(self):
        self.frame_idx+=1
        

class SaveComparisonDatasets(SaveResults):
    def __init__(self, output, gazebo_dataset, folder):
        self.output_folder = f'/home/danc/datasets/localbot/mvg/comparison/{output}'
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)
        else:
            print(f'{Fore.RED} {output} already exists... Aborting SaveComparisonDatasets initialization! {Fore.RESET}')
            exit(0)
            
        dt_now = datetime.now() # current date and time
        config = {'user'       : os.environ["USER"],
                  'date'       : dt_now.strftime("%d/%m/%Y, %H:%M:%S"),
                  'gazebo_dataset' : gazebo_dataset,
                  'folder'   : folder}
        
        with open(f'{self.output_folder}/config.yaml', 'w') as file:
            yaml.dump(config, file)
        
        self.frame_idx = 0 # make sure to save as 00000
        self.csv = pd.DataFrame(columns=('frame', 'position_error (m)', 'rotation_error (rads)'))
        
        print('SaveResults initialized properly')
        
class CompareDatasets(SaveResults):
    def __init__(self, dataset1, dataset2):
       
        self.output_folder = f'{os.environ["HOME"]}/datasets/localbot/dataset_comparison/{dataset1.seq}_and_{dataset2.seq}'
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)
        else:
            print(f'{Fore.RED} {self.output_folder} already exists... Aborting SaveComparisonDatasets initialization! {Fore.RESET}')
            exit(0)
            
        dt_now = datetime.now() # current date and time
        config = {'user'       : os.environ["USER"],
                  'date'       : dt_now.strftime("%d/%m/%Y, %H:%M:%S"),
                  'dataset1'   : dataset1.seq,
                  'dataset2'   : dataset2.seq}
        
        with open(f'{self.output_folder}/config.yaml', 'w') as file:
            yaml.dump(config, file)
        
        self.frame_idx = 0 # make sure to save as 00000
        self.csv = pd.DataFrame(columns=('frame', 'position_error (m)', 'rotation_error (rads)'))
        
        print('CompareDatasets initialized properly')
        
        

        