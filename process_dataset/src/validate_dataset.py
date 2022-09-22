from utils import projectToCamera
import numpy as np
import os
import shutil
from dataset import LocalBotDataset
from sensor_msgs.msg import PointField
import sensor_msgs.point_cloud2 as pc2
from utils import *
import random
from os.path import exists
import yaml
from colorama import Fore
import math
import copy

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
        files = copy.deepcopy(self.files)
        
        config = dataset.getConfig()
        if config['fast']:
            files = ['.rgb.png','.pose.txt']
        else:
            files = copy.deepcopy(self.files)
            files.append('.depth.png')
        
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
        
        # if not (dataset1.getConfig()['is_valid'] and dataset2.getConfig()['is_valid']):
        #     print(f'{Fore.RED} The datasets are not valid! Validate before merge. {Fore.RESET}')
        #     return False
        
        # if not (dataset1.getConfig()['npoints'] == dataset2.getConfig()['npoints']):
        #     print(f'{Fore.RED} The datasets dont have the same number of points! {Fore.RESET}')
        #     return False

        # if not (dataset1.getConfig()['scaled'] == dataset2.getConfig()['scaled']):
        #     print(f'{Fore.RED} Property scaled is different! {Fore.RESET}')
        #     return False
        
        config = dataset1.getConfig()
        if config['fast']:
            files = ['.rgb.png','.pose.txt']
        else:
            files = self.files
        
        size_dataset1 = len(dataset1)
        
        shutil.copytree(dataset1.path_seq, f'{dataset1.root}/{dataset3_name}')
        shutil.copytree(dataset2.path_seq, f'{dataset2.path_seq}_tmp')
        
        dataset3 = LocalBotDataset(path_seq=f'{dataset3_name}')
        dataset2_tmp = LocalBotDataset(path_seq=f'{dataset2.seq}_tmp')
        
        for idx in range(len(dataset2_tmp)):
            for file in files:
                os.rename(f'{dataset2_tmp.path_seq}/frame-{idx:05d}{file}', f'{dataset3.path_seq}/frame-{idx+size_dataset1:05d}{file}')
                
        shutil.rmtree(dataset2_tmp.path_seq)
        
    def createDepthImages(self, dataset, rescale):
        
        # loop through all point clouds
        config = dataset.getConfig()
        
        intrinsic = np.loadtxt(f'{dataset.path_seq}/depth_intrinsic.txt', delimiter=',')
        width = config['depth']['width']
        height = config['depth']['height']
        
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
            
            range_sparse = cv2.resize(range_sparse, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_NEAREST)
            
    
            # Computing the dense depth map
            print('Computing inpaint ...')
            range_dense = cv2.inpaint(range_sparse, mask, 3, cv2.INPAINT_NS)
            print('Inpaint done')
            
            range_dense = cv2.resize(range_dense, (0, 0), fx=1 / rescale, fy=1 / rescale, interpolation=cv2.INTER_NEAREST)
                        
            tmp = copy.deepcopy(range_dense)
            tmp = tmp * 1000.0  # to milimeters
            tmp = tmp.astype(np.uint16)
            cv2.imwrite(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', tmp)
            print(f'Saved depth image {dataset.path_seq}/frame-{idx:05d}.depth.png')
                    
    def createStatistics(self, dataset):
        
        # loop through all point clouds
        config = dataset.getConfig()
        
        
        config['statistics'] = {'B' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))},
                                'G' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))},
                                'R' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))},
                                'D' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))}}
        
        for idx in range(len(dataset)):
            
            print(f'creating stats of frame {idx}')
            
            # Load RGB image
            cv_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.rgb.png', cv2.IMREAD_UNCHANGED)
            
            #cv2.imshow('fig', cv_image)
            #cv2.waitKey(0)
            
            #print(cv_image.shape)
            
            blue_image = cv_image[:,:,0]/255
            green_image = cv_image[:,:,1]/255
            red_image = cv_image[:,:,2]/255
            
            # cv2.imshow('fig', green_image)
            # cv2.waitKey(0)
            
            ## B channel
            config['statistics']['B']['max'][idx] = np.max(blue_image)
            config['statistics']['B']['min'][idx] = np.min(blue_image)
            config['statistics']['B']['mean'][idx] = np.mean(blue_image)
            config['statistics']['B']['std'][idx] = np.std(blue_image)
            
            ## G channel
            config['statistics']['G']['max'][idx] = np.max(green_image)
            config['statistics']['G']['min'][idx] = np.min(green_image)
            config['statistics']['G']['mean'][idx] = np.mean(green_image)
            config['statistics']['G']['std'][idx] = np.std(green_image)
            
            ## R channel
            config['statistics']['R']['max'][idx] = np.max(red_image)
            config['statistics']['R']['min'][idx] = np.min(red_image)
            config['statistics']['R']['mean'][idx] = np.mean(red_image)
            config['statistics']['R']['std'][idx] = np.std(red_image)
            
            
            # Load Depth image
            
            if not config['fast']:                    
                depth_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', cv2.IMREAD_UNCHANGED)
                depth_image = depth_image.astype(np.float32) / 1000.0  # to meters
            else:
                depth_image = -1
            
            ## D channel
            config['statistics']['D']['max'][idx] = np.max(depth_image)
            config['statistics']['D']['min'][idx] = np.min(depth_image)
            config['statistics']['D']['mean'][idx] = np.mean(depth_image)
            config['statistics']['D']['std'][idx] = np.std(depth_image)
                        
        
        config['statistics']['B']['max']  = round(float(np.mean(config['statistics']['B']['max'])),5)
        config['statistics']['B']['min']  = round(float(np.mean(config['statistics']['B']['min'])),5)
        config['statistics']['B']['mean'] = round(float(np.mean(config['statistics']['B']['mean'])),5)
        config['statistics']['B']['std']  = round(float(np.mean(config['statistics']['B']['std'])),5)
        
        config['statistics']['G']['max']  = round(float(np.mean(config['statistics']['G']['max'])),5)
        config['statistics']['G']['min']  = round(float(np.mean(config['statistics']['G']['min'])),5)
        config['statistics']['G']['mean'] = round(float(np.mean(config['statistics']['G']['mean'])),5)
        config['statistics']['G']['std']  = round(float(np.mean(config['statistics']['G']['std'])),5)
    
        config['statistics']['R']['max']  = round(float(np.mean(config['statistics']['R']['max'])),5)
        config['statistics']['R']['min']  = round(float(np.mean(config['statistics']['R']['min'])),5)
        config['statistics']['R']['mean'] = round(float(np.mean(config['statistics']['R']['mean'])),5)
        config['statistics']['R']['std']  = round(float(np.mean(config['statistics']['R']['std'])),5)
        
        config['statistics']['D']['max']  = round(float(np.mean(config['statistics']['D']['max'])),5)
        config['statistics']['D']['min']  = round(float(np.mean(config['statistics']['D']['min'])),5)
        config['statistics']['D']['mean'] = round(float(np.mean(config['statistics']['D']['mean'])),5)
        config['statistics']['D']['std']  = round(float(np.mean(config['statistics']['D']['std'])),5)
        
        dataset.setConfig(config)

    def createStatisticsRGB01(self, dataset):
            
            # loop through all point clouds
            config = dataset.getConfig()
            
            config['statistics'] = {'B' : {'max'  : np.empty((len(dataset))),
                                        'min'  : np.empty((len(dataset))),
                                        'mean' : np.empty((len(dataset))),
                                        'std'  : np.empty((len(dataset)))},
                                    'G' : {'max'  : np.empty((len(dataset))),
                                        'min'  : np.empty((len(dataset))),
                                        'mean' : np.empty((len(dataset))),
                                        'std'  : np.empty((len(dataset)))},
                                    'R' : {'max'  : np.empty((len(dataset))),
                                        'min'  : np.empty((len(dataset))),
                                        'mean' : np.empty((len(dataset))),
                                        'std'  : np.empty((len(dataset)))}}
            
            for idx in range(len(dataset)):
                
                print(f'creating stats of frame {idx}')
                
                # Load RGB image
                cv_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.rgb.png', cv2.IMREAD_UNCHANGED)
                
                #cv2.imshow('fig', cv_image)
                #cv2.waitKey(0)
                
                #print(cv_image.shape)
                
                blue_image = cv_image[:,:,0]/255
                green_image = cv_image[:,:,1]/255
                red_image = cv_image[:,:,2]/255
                
                # cv2.imshow('fig', green_image)
                # cv2.waitKey(0)
                
                ## B channel
                config['statistics']['B']['max'][idx] = np.max(blue_image)
                config['statistics']['B']['min'][idx] = np.min(blue_image)
                config['statistics']['B']['mean'][idx] = np.mean(blue_image)
                config['statistics']['B']['std'][idx] = np.std(blue_image)
                
                ## G channel
                config['statistics']['G']['max'][idx] = np.max(green_image)
                config['statistics']['G']['min'][idx] = np.min(green_image)
                config['statistics']['G']['mean'][idx] = np.mean(green_image)
                config['statistics']['G']['std'][idx] = np.std(green_image)
                
                ## R channel
                config['statistics']['R']['max'][idx] = np.max(red_image)
                config['statistics']['R']['min'][idx] = np.min(red_image)
                config['statistics']['R']['mean'][idx] = np.mean(red_image)
                config['statistics']['R']['std'][idx] = np.std(red_image)
                
            
                            
            
            config['statistics']['B']['max'] = round(float(np.mean(config['statistics']['B']['max'])),5)
            config['statistics']['B']['min'] = round(float(np.mean(config['statistics']['B']['min'])),5)
            config['statistics']['B']['mean'] = round(float(np.mean(config['statistics']['B']['mean'])),5)
            config['statistics']['B']['std'] = round(float(np.mean(config['statistics']['B']['std'])),5)
            
            config['statistics']['G']['max'] = round(float(np.mean(config['statistics']['G']['max'])),5)
            config['statistics']['G']['min'] = round(float(np.mean(config['statistics']['G']['min'])),5)
            config['statistics']['G']['mean'] = round(float(np.mean(config['statistics']['G']['mean'])),5)
            config['statistics']['G']['std'] = round(float(np.mean(config['statistics']['G']['std'])),5)
        
            config['statistics']['R']['max'] = round(float(np.mean(config['statistics']['R']['max'])),5)
            config['statistics']['R']['min'] = round(float(np.mean(config['statistics']['R']['min'])),5)
            config['statistics']['R']['mean'] = round(float(np.mean(config['statistics']['R']['mean'])),5)
            config['statistics']['R']['std'] = round(float(np.mean(config['statistics']['R']['std'])),5)
            

            dataset.setConfig(config)
    
    def processImages(self, dataset, technique, global_dataset):
        
        config = dataset.getConfig()
        
        config['statistics'] = {'B' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))},
                                'G' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))},
                                'R' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))},
                                'D' : {'max'  : np.empty((len(dataset))),
                                       'min'  : np.empty((len(dataset))),
                                       'mean' : np.empty((len(dataset))),
                                       'std'  : np.empty((len(dataset)))}}
        
        
        # Local Processing
        if global_dataset == None:
        
            for idx in range(len(dataset)):
                
                # RGB image
                
                bgr_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.rgb.png', cv2.IMREAD_UNCHANGED)
                blue_image = bgr_image[:,:,0]
                green_image = bgr_image[:,:,1]
                red_image = bgr_image[:,:,2]
                
                depth_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', cv2.IMREAD_UNCHANGED)
                depth_image = depth_image.astype(np.float32) / 1000.0  # to meters
                
                
                if technique =='standardization':
                    
                    # B channel
                    mean = np.mean(blue_image)
                    std = np.std(blue_image)
                    blue_image = (blue_image - mean) / std
                    
                    # G channel
                    mean = np.mean(green_image)
                    std = np.std(green_image)
                    green_image = (green_image - mean) / std
                    
                    # R channel
                    mean = np.mean(red_image)
                    std = np.std(red_image)
                    red_image = (red_image - mean) / std
                    
                    # D channel
                    mean = np.mean(depth_image)
                    std = np.std(depth_image)
                    depth_image = (depth_image - mean) / std

                elif technique == 'normalization':
                    
                    # B channel
                    min_v = np.min(blue_image)
                    max_v = np.max(blue_image)
                    blue_image = (blue_image - min_v) / (max_v - min_v)
                    
                    # G channel
                    min_v = np.min(green_image)
                    max_v = np.max(green_image)
                    green_image = (green_image - min_v) / (max_v - min_v)
                    
                    # R channel
                    min_v = np.min(red_image)
                    max_v = np.max(red_image)
                    red_image = (red_image - min_v) / (max_v - min_v)
                    
                    # D channel
                    min_v = np.min(depth_image)
                    max_v = np.max(depth_image)
                    depth_image = (depth_image - min_v) / (max_v - min_v)
                
                else:
                    print('Tehcnique not implemented. Available techniques are: normalization and standardization')
                    exit(0)
                
                    
                blue_image = blue_image.astype(np.float32)
                green_image = green_image.astype(np.float32)
                red_image = red_image.astype(np.float32)
                depth_image = depth_image.astype(np.float32)
                
                
                
                ## B channel
                config['statistics']['B']['max'][idx] = np.max(blue_image)
                config['statistics']['B']['min'][idx] = np.min(blue_image)
                config['statistics']['B']['mean'][idx] = np.mean(blue_image)
                config['statistics']['B']['std'][idx] = np.std(blue_image)
                
                ## G channel
                config['statistics']['G']['max'][idx] = np.max(green_image)
                config['statistics']['G']['min'][idx] = np.min(green_image)
                config['statistics']['G']['mean'][idx] = np.mean(green_image)
                config['statistics']['G']['std'][idx] = np.std(green_image)
                
                ## R channel
                config['statistics']['R']['max'][idx] = np.max(red_image)
                config['statistics']['R']['min'][idx] = np.min(red_image)
                config['statistics']['R']['mean'][idx] = np.mean(red_image)
                config['statistics']['R']['std'][idx] = np.std(red_image)
                
                ## D channel
                config['statistics']['D']['max'][idx] = np.max(depth_image)
                config['statistics']['D']['min'][idx] = np.min(depth_image)
                config['statistics']['D']['mean'][idx] = np.mean(depth_image)
                config['statistics']['D']['std'][idx] = np.std(depth_image)
                
                # joint BGR images as nparray
                
                bgr_image = cv2.merge([blue_image, green_image, red_image])
                
                os.remove(f'{dataset.path_seq}/frame-{idx:05d}.depth.png')
                os.remove(f'{dataset.path_seq}/frame-{idx:05d}.rgb.png')
                
                np.save(f'{dataset.path_seq}/frame-{idx:05d}.depth.npy', depth_image)
                np.save(f'{dataset.path_seq}/frame-{idx:05d}.rgb.npy', bgr_image)
                #cv2.imwrite(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', tmp)

            config['processing'] = {'global'    : None,
                                    'technique' : technique}
            
            config['statistics']['B']['max'] = round(float(np.mean(config['statistics']['B']['max'])),5)
            config['statistics']['B']['min'] = round(float(np.mean(config['statistics']['B']['min'])),5)
            config['statistics']['B']['mean'] = round(float(np.mean(config['statistics']['B']['mean'])),5)
            config['statistics']['B']['std'] = round(float(np.mean(config['statistics']['B']['std'])),5)
            
            config['statistics']['G']['max'] = round(float(np.mean(config['statistics']['G']['max'])),5)
            config['statistics']['G']['min'] = round(float(np.mean(config['statistics']['G']['min'])),5)
            config['statistics']['G']['mean'] = round(float(np.mean(config['statistics']['G']['mean'])),5)
            config['statistics']['G']['std'] = round(float(np.mean(config['statistics']['G']['std'])),5)
        
            config['statistics']['R']['max'] = round(float(np.mean(config['statistics']['R']['max'])),5)
            config['statistics']['R']['min'] = round(float(np.mean(config['statistics']['R']['min'])),5)
            config['statistics']['R']['mean'] = round(float(np.mean(config['statistics']['R']['mean'])),5)
            config['statistics']['R']['std'] = round(float(np.mean(config['statistics']['R']['std'])),5)
            
            config['statistics']['D']['max'] = round(float(np.mean(config['statistics']['D']['max'])),5)
            config['statistics']['D']['min'] = round(float(np.mean(config['statistics']['D']['min'])),5)
            config['statistics']['D']['mean'] = round(float(np.mean(config['statistics']['D']['mean'])),5)
            config['statistics']['D']['std'] = round(float(np.mean(config['statistics']['D']['std'])),5)
            
            dataset.setConfig(config)
        
        # Global Processing
        else:
            global_config = global_dataset.getConfig()
            
            global_stats = global_config['statistics']
            
            for idx in range(len(dataset)):
                
                bgr_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.rgb.png', cv2.IMREAD_UNCHANGED)
                blue_image = bgr_image[:,:,0]
                green_image = bgr_image[:,:,1]
                red_image = bgr_image[:,:,2]
                
                depth_image = cv2.imread(f'{dataset.path_seq}/frame-{idx:05d}.depth.png', cv2.IMREAD_UNCHANGED)
                depth_image = depth_image.astype(np.float32) / 1000.0  # to meters
                
                
                if technique =='standardization':
                    
                    blue_image = (blue_image - global_stats['B']['mean']) /  global_stats['B']['std']
                    green_image = (green_image - global_stats['G']['mean']) /  global_stats['G']['std']
                    red_image = (red_image - global_stats['R']['mean']) /  global_stats['R']['std']
                    depth_image = (depth_image - global_stats['D']['mean']) /  global_stats['D']['std']

                elif technique == 'normalization':
                    blue_image = (blue_image - global_stats['B']['min']) / (global_stats['B']['max'] - global_stats['B']['min'])
                    green_image = (green_image - global_stats['G']['min']) / (global_stats['G']['max'] - global_stats['G']['min'])
                    red_image = (red_image - global_stats['R']['min']) / (global_stats['R']['max'] - global_stats['R']['min'])
                    depth_image = (depth_image - global_stats['D']['min']) / (global_stats['D']['max'] - global_stats['D']['min'])
                    
                else:
                    print('Tehcnique not implemented. Available techniques are: normalization and standardization')
                    exit(0)
                    
                blue_image = blue_image.astype(np.float32)
                green_image = green_image.astype(np.float32)
                red_image = red_image.astype(np.float32)
                depth_image = depth_image.astype(np.float32)
                
                ## B channel
                config['statistics']['B']['max'][idx] = np.max(blue_image)
                config['statistics']['B']['min'][idx] = np.min(blue_image)
                config['statistics']['B']['mean'][idx] = np.mean(blue_image)
                config['statistics']['B']['std'][idx] = np.std(blue_image)
                
                ## G channel
                config['statistics']['G']['max'][idx] = np.max(green_image)
                config['statistics']['G']['min'][idx] = np.min(green_image)
                config['statistics']['G']['mean'][idx] = np.mean(green_image)
                config['statistics']['G']['std'][idx] = np.std(green_image)
                
                ## R channel
                config['statistics']['R']['max'][idx] = np.max(red_image)
                config['statistics']['R']['min'][idx] = np.min(red_image)
                config['statistics']['R']['mean'][idx] = np.mean(red_image)
                config['statistics']['R']['std'][idx] = np.std(red_image)
                
                ## D channel
                config['statistics']['D']['max'][idx] = np.max(depth_image)
                config['statistics']['D']['min'][idx] = np.min(depth_image)
                config['statistics']['D']['mean'][idx] = np.mean(depth_image)
                config['statistics']['D']['std'][idx] = np.std(depth_image)
                
                # joint BGR images as nparray
                bgr_image = cv2.merge([blue_image, green_image, red_image])
                
                os.remove(f'{dataset.path_seq}/frame-{idx:05d}.depth.png')
                os.remove(f'{dataset.path_seq}/frame-{idx:05d}.rgb.png')
                
                np.save(f'{dataset.path_seq}/frame-{idx:05d}.depth.npy', depth_image)
                np.save(f'{dataset.path_seq}/frame-{idx:05d}.rgb.npy', bgr_image)

            config['processing'] = {'global'    : global_dataset.path_seq,
                                    'technique' : technique}
            
            config['statistics']['B']['max'] = round(float(np.mean(config['statistics']['B']['max'])),5)
            config['statistics']['B']['min'] = round(float(np.mean(config['statistics']['B']['min'])),5)
            config['statistics']['B']['mean'] = round(float(np.mean(config['statistics']['B']['mean'])),5)
            config['statistics']['B']['std'] = round(float(np.mean(config['statistics']['B']['std'])),5)
            
            config['statistics']['G']['max'] = round(float(np.mean(config['statistics']['G']['max'])),5)
            config['statistics']['G']['min'] = round(float(np.mean(config['statistics']['G']['min'])),5)
            config['statistics']['G']['mean'] = round(float(np.mean(config['statistics']['G']['mean'])),5)
            config['statistics']['G']['std'] = round(float(np.mean(config['statistics']['G']['std'])),5)
        
            config['statistics']['R']['max'] = round(float(np.mean(config['statistics']['R']['max'])),5)
            config['statistics']['R']['min'] = round(float(np.mean(config['statistics']['R']['min'])),5)
            config['statistics']['R']['mean'] = round(float(np.mean(config['statistics']['R']['mean'])),5)
            config['statistics']['R']['std'] = round(float(np.mean(config['statistics']['R']['std'])),5)
            
            config['statistics']['D']['max'] = round(float(np.mean(config['statistics']['D']['max'])),5)
            config['statistics']['D']['min'] = round(float(np.mean(config['statistics']['D']['min'])),5)
            config['statistics']['D']['mean'] = round(float(np.mean(config['statistics']['D']['mean'])),5)
            config['statistics']['D']['std'] = round(float(np.mean(config['statistics']['D']['std'])),5)
            
            dataset.setConfig(config)
            
            
            
    
            
            
        
        
            
            
        
        
        
        
        
    