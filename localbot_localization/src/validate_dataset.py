from turtle import st
from cv2 import TermCriteria_COUNT
import torch.utils.data as data
from localbot_localization.src.utilities import normalize_quat, projectToCamera
import numpy as np
import torch
import os
import shutil
from localbot_localization.src.dataset import LocalBotDataset
import localbot_core.src.pypcd as pypcd
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from localbot_core.src.utilities import *
import random
from os.path import exists
import yaml
from colorama import Fore
import math

class ValidateDataset():
    def __init__(self):
        self.files = ['.pcd', '.rgb.png', '.pose.txt']
        
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
    
    def invalidFrames(self, dataset):
        # return a list with invalid frames
        idxs = []
        for index in range(len(dataset)):
            for file in self.files:
                if not exists(f'{dataset.path_seq}/frame-{index:05d}{file}'):
                    idxs.append(index)
                    break
        return idxs
                    
    def removeFrames(self, dataset, idxs):
        for idx in idxs:
            for file in os.listdir(f'{dataset.path_seq}'):
                if file.startswith(f'frame-{idx:05d}'):
                    os.remove(f'{dataset.path_seq}/{file}')
            
                    
    def reorganizeDataset(self, dataset):
        # here I assume the invalidFrames and removeFrames were called before.
        # last_pose_idx is the idx of the last frame. We cannot use len(dataset) because the dataset might be missing some frames!
        last_pose_idx = int(sorted([f for f in os.listdir(dataset.path_seq) if f.endswith('pose.txt')])[-1][6:11])
        
        for idx in range(last_pose_idx+1):
            print(idx)
            if not exists(f'{dataset.path_seq}/frame-{idx:05d}.pose.txt'):
                # idx does not exists, so we have to rename the close one.
                print(f'{idx} is missing!!!')
                new_idx = None
                for idx2 in range(idx+1, last_pose_idx+1):
                    print(f'trying {idx2}')
                    if exists(f'{dataset.path_seq}/frame-{idx2:05d}.pose.txt'):
                        new_idx = idx2
                        break
                if not new_idx==None:    
                    print(f'renaming idx {new_idx} to idx {idx}')
                    for file in self.files:
                        os.rename(f'{dataset.path_seq}/frame-{new_idx:05d}{file}', f'{dataset.path_seq}/frame-{idx:05d}{file}')
                else:
                    print(f'No candidate to replace {idx}')
                    
    
    def validateDataset(self, dataset):
        # update config, check if all point clouds have the same size, if any has nans
        
        # check for invalid frames
        idxs = self.invalidFrames(dataset)
        if idxs != []:
            print(f'{Fore.RED} There are invalid frames in the dataset! {Fore.RESET}')
            return False

        # # check for missing data
        # dct = self.numberOfNans(dataset)
        # n_nans = 0
        # for count in dct.values():
        #     n_nans += count
        # if n_nans != 0:
        #     print(f'{Fore.RED} There are Nans in the dataset! {Fore.RESET}')
        #     return False 
        
        # #check for point clouds of different size
        # dct = self.numberOfPoints(dataset)
        # number_of_points = list(dct.values())
        # result = all(element == number_of_points[0] for element in number_of_points)
        # if not result:
        #     print(f'{Fore.RED} Not all pointclouds have the same number of points! {Fore.RESET}')
        #     return False
        
        config = dataset.getConfig()
        config['is_valid'] = True
        with open(f'{dataset.path_seq}/config.yaml', 'w') as file:
            yaml.dump(config, file)
        
        return True

    def mergeDatasets(self, dataset1, dataset2, dataset3_name):
        # both datasets must be valids
        # they should share the same number of points
        
        if not (dataset1.getConfig()['is_valid'] and dataset2.getConfig()['is_valid']):
            print(f'{Fore.RED} The datasets are not valid! Validate before merge. {Fore.RESET}')
            return False
        
        if not (dataset1.getConfig()['npoints'] == dataset2.getConfig()['npoints']):
            print(f'{Fore.RED} The datasets dont have the same number of points! {Fore.RESET}')
            return False

        if not (dataset1.getConfig()['scaled'] == dataset2.getConfig()['scaled']):
            print(f'{Fore.RED} Property scaled is different! {Fore.RESET}')
            return False
        
        size_dataset1 = len(dataset1)
        
        shutil.copytree(dataset1.path_seq, f'{dataset1.root}/{dataset3_name}')
        shutil.copytree(dataset2.path_seq, f'{dataset2.path_seq}_tmp')
        
        dataset3 = LocalBotDataset(path_seq=f'{dataset3_name}')
        dataset2_tmp = LocalBotDataset(path_seq=f'{dataset2.seq}_tmp')
        
        for idx in range(len(dataset2_tmp)):
            for file in self.files:
                os.rename(f'{dataset2_tmp.path_seq}/frame-{idx:05d}{file}', f'{dataset3.path_seq}/frame-{idx+size_dataset1:05d}{file}')
                
        shutil.rmtree(dataset2_tmp.path_seq)
        
    def createDepthImages(self, dataset, size):
        
        # loop through all point clouds
        config = dataset.getConfig()
        
        intrinsic = np.loadtxt(f'{dataset.path_seq}/depth_intrinsic.txt', delimiter=',')
        width = dataset.getConfig()['depth']['width']
        height = dataset.getConfig()['depth']['height']
        
        max_array = np.empty((len(dataset)))
        min_array = np.empty((len(dataset)))
        mean_array = np.empty((len(dataset)))
        std_array = np.empty((len(dataset)))
        
        for idx in range(len(dataset)):
            pc_raw = read_pcd(f'{dataset.path_seq}/frame-{idx:05d}.pcd')
            pts = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z'], pc_raw.pc_data['intensity']])  # stays 4xN
            
            pixels, valid_pixels, dist = projectToCamera(intrinsic, [0, 0, 0, 0, 0], width, height, pts)
            
            range_sparse = np.zeros((height, width), dtype=np.float32)
            mask = 255 * np.ones((range_sparse.shape[0], range_sparse.shape[1]), dtype=np.uint8)
        
        
            for idx_point in range(0, pts.shape[1]):
                if valid_pixels[idx_point]:
                    x0 = math.floor(pixels[0, idx_point])
                    y0 = math.floor(pixels[1, idx_point])
                    mask[y0, x0] = 0
                    range_sparse[y0, x0] = dist[idx_point]
                    
            range_sparse = cv2.resize(range_sparse, (size,size), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            
            # Computing the dense depth map
            print('Computing inpaint ...')
            range_dense = cv2.inpaint(range_sparse, mask, 3, cv2.INPAINT_NS)
            print('Inpaint done')
            
            min_array[idx] = np.min(range_dense)
            max_array[idx] = np.max(range_dense)
            mean_array[idx] = np.mean(range_dense)
            std_array[idx] = np.std(range_dense)
                        
            tmp = copy.deepcopy(range_dense)
            tmp = tmp * 1000.0  # to milimeters
            tmp = tmp.astype(np.uint16)
            cv2.imwrite(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', tmp)
            print(f'Saved depth image {dataset.path_seq}/frame-{idx:05d}.depth.png')
            
        config['depth']['statistics'] = {}
        config['depth']['statistics']['max'] = round(float(np.mean(max_array)),5)
        config['depth']['statistics']['min'] = round(float(np.mean(min_array)),5)
        config['depth']['statistics']['mean'] = round(float(np.mean(mean_array)),5)
        config['depth']['statistics']['std'] = round(float(np.mean(std_array)),5)
        
        dataset.setConfig(config)
    
    def processDepthImages(self, dataset, technique, global_dataset):
        
        config = dataset.getConfig()
        
        max_array = np.empty((len(dataset)))
        min_array = np.empty((len(dataset)))
        mean_array = np.empty((len(dataset)))
        std_array = np.empty((len(dataset)))
        
        
        # Local Processing
        if global_dataset == None:
        
            for idx in range(len(dataset)):
                cv_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', cv2.IMREAD_UNCHANGED)
                cv_image = cv_image.astype(np.float32) / 1000.0  # to meters
                
                if technique =='standardization':
                    mean = np.mean(cv_image)
                    std = np.std(cv_image)
                    final_cv_image = (cv_image - mean) / std

                elif technique == 'normalization':
                    min_v = np.min(cv_image)
                    max_v = np.max(cv_image)
                    final_cv_image = (cv_image - min_v) / (max_v - min_v)
                
                else:
                    print('Tehcnique not implemented. Available techniques are: normalization and standardization')
                    exit(0)
                    
                tmp = copy.deepcopy(final_cv_image).astype(np.float32)
                
                min_array[idx] = np.min(tmp)
                max_array[idx] = np.max(tmp)
                mean_array[idx] = np.mean(tmp)
                std_array[idx] = np.std(tmp)
                
                
                os.remove(f'{dataset.path_seq}/frame-{idx:05d}.depth.png')
                np.save(f'{dataset.path_seq}/frame-{idx:05d}.depth.npy', tmp)
                #cv2.imwrite(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', tmp)

            config['depth']['preprocessing'] = {'global'    : None,
                                                'technique' : technique}
            config['depth']['statistics']['max'] = round(float(np.mean(max_array)),5)
            config['depth']['statistics']['min'] = round(float(np.mean(min_array)),5)
            config['depth']['statistics']['mean'] = round(float(np.mean(mean_array)),5)
            config['depth']['statistics']['std'] = round(float(np.mean(std_array)),5)
            dataset.setConfig(config)
        
        # Global Processing
        else:
            global_config = global_dataset.getConfig()
            max_global = global_config['depth']['statistics']['max']
            min_global = global_config['depth']['statistics']['min']
            mean_global = global_config['depth']['statistics']['mean']
            std_global = global_config['depth']['statistics']['std']
            
            for idx in range(len(dataset)):
                cv_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', cv2.IMREAD_UNCHANGED)
                cv_image = cv_image.astype(np.float32) / 1000.0  # to meters
                
                if technique =='standardization':
                    final_cv_image = (cv_image - mean_global) / std_global

                elif technique == 'normalization':
                    final_cv_image = (cv_image - min_global) / (max_global - min_global)
                else:
                    print('Tehcnique not implemented. Available techniques are: normalization and standardization')
                    exit(0)
                    
                tmp = copy.deepcopy(final_cv_image).astype(np.float32)
                
                min_array[idx] = np.min(tmp)
                max_array[idx] = np.max(tmp)
                mean_array[idx] = np.mean(tmp)
                std_array[idx] = np.std(tmp)
                
                os.remove(f'{dataset.path_seq}/frame-{idx:05d}.depth.png')
                np.save(f'{dataset.path_seq}/frame-{idx:05d}.depth.npy', tmp)
                #cv2.imwrite(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', tmp)

            config['depth']['preprocessing'] = {'global'    : global_dataset.path_seq,
                                                'technique' : technique}
            config['depth']['statistics']['max'] = round(float(np.mean(max_array)),5)
            config['depth']['statistics']['min'] = round(float(np.mean(min_array)),5)
            config['depth']['statistics']['mean'] = round(float(np.mean(mean_array)),5)
            config['depth']['statistics']['std'] = round(float(np.mean(std_array)),5)
            dataset.setConfig(config)
            
            
            
    def globalProcessDepthImages(self, dataset, technique):
        
        config = dataset.getConfig()
        
        for idx in range(len(dataset)):
            cv_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', cv2.IMREAD_UNCHANGED)
            cv_image = cv_image.astype(np.float32) / 1000.0  # to meters
            
            if technique =='standardization':
                mean = np.mean(cv_image)
                std = np.std(cv_image)
                final_cv_image = (cv_image - mean) / std

            elif technique == 'normalization':
                min_v = np.min(cv_image)
                max_v = np.max(cv_image)
                final_cv_image = (cv_image - min_v) / (max_v - min_v)
            
            else:
                print('Tehcnique not implemented. Available techniques are: normalization and standardization')
                exit(0)
                
            tmp = copy.deepcopy(final_cv_image).astype(np.float32)
            
            os.remove(f'{dataset.path_seq}/frame-{idx:05d}.depth.png')
            np.save(f'{dataset.path_seq}/frame-{idx:05d}.depth.npy', tmp)
            #cv2.imwrite(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', tmp)

        config['depth']['preprocessing'] = {'global' : None,
                                            'technique' : technique}
        dataset.setConfig(config)
            

            
    def globalProcessDepthImages(self, dataset, type):
        
        config=0
        technique=0
        
        config['depth']['preprocessing'] = {'global' : {'dataset_train':None,
                                                        'max' : None,
                                                        'min' : None,
                                                        'mean': None,
                                                        'std' : None },
                                            'technique' : technique}
        
        for idx in range(len(dataset)):
            cv_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', cv2.IMREAD_UNCHANGED)
            cv_image = cv_image.astype(np.float32) / 1000.0  # to meters
            
            mean = np.mean(cv_image)
            std = np.std(cv_image)
            
            #print(f'mean: {mean}')
            #print(f'std: {std}')
            
            if type =='standardization':
                final_cv_image = (cv_image - mean) / std
                
                print(final_cv_image.shape)
                
                cv2.imshow('image', final_cv_image)
                cv2.waitKey(0)
            elif type == 'normalization':
                min_v = np.min(cv_image)
                max_v = np.max(cv_image)
                
                final_cv_image = (cv_image - min_v) / (max_v - min_v)
                cv2.imshow('image', final_cv_image)
                cv2.waitKey(0)
            
            
        

                                                
            
            
            
        
        
            
            
        
        
        
        
        
    
