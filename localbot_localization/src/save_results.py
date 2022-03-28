#!/usr/bin/env python3

# 3rd-party

from operator import pos
import rospy
import os
from visualization_msgs.msg import *
from cv_bridge import CvBridge
from tf.listener import TransformListener
from localbot_core.src.utilities import *
from sensor_msgs.msg import PointCloud2, Image, PointField
from colorama import Fore
from datetime import datetime
import yaml
import pandas as pd

class SaveResults():
    """
    class to save results
    """
    
    def __init__(self, output, model_path, seq):
        
        # attribute initializer
        self.output_folder = f'{os.environ["HOME"]}/results/localbot/{output}'
        self.model_path = model_path
        self.seq = seq
        
        if not os.path.exists(self.output_folder):
            print(f'Creating folder {self.output_folder}')
            os.makedirs(self.output_folder)  # Create the new folder
        
        else:
            print(f'{Fore.RED} {self.output_folder} already exists... Aborting SaveResults initialization! {Fore.RESET}')
            exit(0)
        
        
        dt_now = datetime.now() # current date and time
        config = {'user' : os.environ["USER"],
                  'date' : dt_now.strftime("%d/%m/%Y, %H:%M:%S"),
                  'model_path' : self.model_path,
                  'seq' : f'seq{self.seq}'}
        
        with open(f'{self.output_folder}/log.yaml', 'w') as file:
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
                           'position_error' : position_error,
                           'rotation_error' : rotation_error}
        
        self.csv = self.csv.append(row, ignore_index=True)  
        
    def saveCSV(self):
        # save averages values in the last row
        row = {'frame' : 'average_values', 
                         'position_error' : self.csv.mean(axis=0).loc["position_error"],
                         'rotation_error' : self.csv.mean(axis=0).loc["rotation_error"]}
        self.csv = self.csv.append(row, ignore_index=True)  
        print(self.csv)
        self.csv.to_csv(f'{self.output_folder}/errors.csv', index=False, float_format='%.5f')

    def step(self):
        self.frame_idx+=1
        print(f'Moving to {self.frame_idx:05d}')
        
        

        