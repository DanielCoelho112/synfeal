import torch.utils.data as data
from localbot_localization.src.utilities import normalize_quat
import numpy as np
import torch
import os
import shutil
from localbot_localization.src.dataset import LocalBotDataset
import localbot_core.src.pypcd as pypcd
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from localbot_core.src.utilities import *
import random
from os.path import exists
import yaml

class ValidateDataset():
    def __init__(self):
        self.config = {}
        
    def resetConfig(self, config={}):
        self.config = config
    
    def duplicateDataset(self, dataset, suffix):
        # copy folder and create dataset object, return object
        path_dataset = dataset.path_seq
        path_validated_dataset = f'{dataset.path_seq}{suffix}'
        shutil.copytree(path_dataset, path_validated_dataset)
        return LocalBotDataset(path_seq=f'{dataset.seq}{suffix}')
    
    def numberOfPoints(self, dataset, frame = None):
        dct = {}
        if frame == None: # calculate number of points for all pointclouds
            for index in range(len(dataset)):
                n_points = read_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd').points
                dct[index] = n_points
        else:
            n_points = read_pcd(f'{dataset.path_seq}/frame-{frame:05d}.pcd').points
            dct[frame] = n_points
        return dct
        
    def numberOfNans(self, dataset, frame = None):
        dct = {}
        if frame == None:
            for index in range(len(dataset)):
                pc_raw = read_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd')
                pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z']]).T  # stays NX3
                dct[index] = np.sum(np.isnan(pts).any(axis=1))
        else:
            pc_raw = read_pcd(f'{dataset.path_seq}/frame-{frame:05d}.pcd')
            pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z']]).T  # stays NX3
            dct[frame] = np.sum(np.isnan(pts).any(axis=1))
        return dct
            
    def removeNansPointCloud(self, dataset, frame = None):
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        
        if frame == None:
            for index in range(len(dataset)):
                pc_raw = read_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd')
                pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z'], pc_raw.pc_data['intensity']]).T  # stays NX4
                pts = pts[~np.isnan(pts).any(axis=1)]
        
                # del 
                os.remove(f'{dataset.path_seq}/frame-{index:05d}.pcd')
                # save new point clouds
                pc_msg =pc2.create_cloud(None, fields, pts)
                write_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd', pc_msg)
        else:
            pc_raw = read_pcd(f'{dataset.path_seq}/frame-{frame:05d}.pcd')
            pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z'], pc_raw.pc_data['intensity']]).T  # stays NX4
            pts = pts[~np.isnan(pts).any(axis=1)]
    
            # del 
            os.remove(f'{dataset.path_seq}/frame-{frame:05d}.pcd')
            # save new point clouds
            pc_msg =pc2.create_cloud(None, fields, pts)
            write_pcd(f'{dataset.path_seq}/frame-{frame:05d}.pcd', pc_msg)
        
              
    def downsamplePointCloud(self, dataset, npoints):
        
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        for index in range(len(dataset)):
            pc_raw = read_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd')
            pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z'], pc_raw.pc_data['intensity']]).T  # stays NX4
            
            initial_npoints = pc_raw.points
            
            step = initial_npoints // npoints
            
            idxs = list(range(0,initial_npoints, step))
            
            for i in range(len(idxs) - npoints):
                idxs.pop(random.randrange(len(idxs)))
                
            pts = pts[idxs,:]
            
            # del 
            os.remove(f'{dataset.path_seq}/frame-{index:05d}.pcd')
            # save new point clouds
            pc_msg =pc2.create_cloud(None, fields, pts)
            write_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd', pc_msg)
        
        config = dataset.getConfig()
        config['npoints'] = npoints
        with open(f'{dataset.path_seq}/config.yaml', 'w') as file:
            yaml.dump(config, file)
            
    def scalePointCloud(self, dataset):
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
        
        for index in range(len(dataset)):
            pc_raw = read_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd')
            pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z'], pc_raw.pc_data['intensity']]).T  # stays NX4
            pts = pts - np.expand_dims(np.mean(pts, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(pts ** 2, axis=1)), 0)
            pts = pts / dist
            os.remove(f'{dataset.path_seq}/frame-{index:05d}.pcd')
            # save new point clouds
            pc_msg =pc2.create_cloud(None, fields, pts)
            write_pcd(f'{dataset.path_seq}/frame-{index:05d}.pcd', pc_msg)

        config = dataset.getConfig()
        config['scaled'] = True
        with open(f'{dataset.path_seq}/config.yaml', 'w') as file:
            yaml.dump(config, file)
    
    def invalidFrames(self, dataset, files = ['.pcd', '.rgb.png', '.pose.txt']):
        # return a list with invalid frames
        idxs = []
        for index in range(len(dataset)):
            for file in files:
                if not exists(f'{dataset.path_seq}/frame-{index:05d}{file}'):
                    idxs.append(index)
                    break
        return idxs
                    
    def removeFrames(self, dataset, idxs):
        for idx in idxs:
            for file in os.listdir(f'{dataset.path_seq}'):
                if file.startswith(f'frame-{idx:05d}'):
                    os.remove(f'{dataset.path_seq}/{file}')
            
                    
    def reorganizeDataset(self, dataset, files = ['.pcd', '.rgb.png', '.pose.txt']):
        # here I assume the invalidFrames and removeFrames were called before.
        for idx in range(len(dataset)):
            if not exists(f'{dataset.path_seq}/frame-{idx:05d}.pose.txt'):
                for file in files:
                    os.rename(f'{dataset.path_seq}/frame-{idx+1:05d}{file}', f'{dataset.path_seq}/frame-{idx:05d}{file}')
                
    
    def validateDataset(self, dataset):
        # update config, check if all point clouds have the same size, if any has nans
        
        # check for invalid frames
        idxs = self.invalidFrames(dataset)
        if idxs != []:
            return False

        # check for missing data
        dct = self.numberOfNans(dataset)
        n_nans = 0
        for count in dct.values():
            n_nans += count
        if n_nans != 0:
            return False 
        
        #check for point clouds of different size
        dct = self.numberOfPoints(dataset)
        number_of_points = list(dct.values())
        result = all(element == number_of_points[0] for element in number_of_points)
        if not result:
            return False
        
        config = dataset.getConfig()
        config['is_valid'] = True
        with open(f'{dataset.path_seq}/config.yaml', 'w') as file:
            yaml.dump(config, file)
        
        return True

    def mergeDatasets(self, dataset1, dataset2):
        pass
    
