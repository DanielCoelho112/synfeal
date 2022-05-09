import cv2
from cv2 import INTER_AREA
import torch.utils.data as data
from localbot_core.src.utilities import read_pcd, matrixToXYZ, matrixToQuaternion
from localbot_localization.src.utilities import normalize_quat
import numpy as np
import torch
import os
import yaml
from yaml.loader import SafeLoader
from PIL import Image
from torchvision import transforms

# pytorch datasets: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class LocalBotDataset(data.Dataset):
    def __init__(self, path_seq, rgb_transform = None, depth_transform = None, inputs = None):
        self.root = f'{os.environ["HOME"]}/datasets/localbot'
        self.seq = path_seq
        self.path_seq = f'{self.root}/{path_seq}'
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        if inputs == None:
            self.inputs = ['point_cloud', 'depth_image', 'rgb_image']
        else:
            self.inputs = inputs
        
        config = self.getConfig()
        if 'statistics' in config:
            self.depth_mean = config['statistics']['D']['mean']
            self.depth_std = config['statistics']['D']['std']
        
        
    def __getitem__(self, index):
        
        output = []
        
        if 'point_cloud' in self.inputs:
            # load point cloud
            pc_raw = read_pcd(f'{self.path_seq}/frame-{index:05d}.pcd')
            point_set = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z']]).T  # stays NX3
            point_set = torch.from_numpy(point_set.astype(np.float32))
            output.append(point_set)
        
        if 'depth_image' in self.inputs:
            # load depth image
            depth_image = cv2.imread(f'{self.path_seq}/frame-{index:05d}.depth.png', cv2.IMREAD_UNCHANGED)
            depth_image = depth_image.astype(np.float32) / 1000.0  # to meters
            depth_image = Image.fromarray(depth_image)
            
            if self.depth_transform!=None:
                depth_image = self.depth_transform(depth_image)
            
            output.append(depth_image)
        
        if 'rgb_image' in self.inputs:
            # TODO: change this to the correct dataset
            rgb_image = Image.open(f'{self.path_seq}/frame-{index:05d}.rgb.png')
            
            if self.rgb_transform != None:
                rgb_image = self.rgb_transform(rgb_image)
            output.append(rgb_image)

        
        # load pose
        matrix = np.loadtxt(f'{self.path_seq}/frame-{index:05d}.pose.txt', delimiter=',')
        quaternion = matrixToQuaternion(matrix)
        quaternion = normalize_quat(quaternion)
        xyz = matrixToXYZ(matrix)
        pose = np.append(xyz, quaternion)
        pose = torch.from_numpy(pose.astype(np.float32))
        output.append(pose)
        
        return tuple(output)
    

    def __len__(self):
        return sum(f.endswith('pose.txt') for f in os.listdir(self.path_seq))
    
    def getConfig(self):
        with open(f'{self.path_seq}/config.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
        return config

    def setConfig(self, config):
        with open(f'{self.path_seq}/config.yaml', 'w') as f:
            yaml.dump(config, f)
      
      
# config_stats = LocalBotDataset('seq5',depth_transform=None ,rgb_transform=None, inputs=['depth_image']).getConfig()['statistics']
# rgb_mean = [config_stats['R']['mean'], config_stats['G']['mean'], config_stats['B']['mean']]
# rgb_std = [config_stats['R']['std'], config_stats['G']['std'], config_stats['B']['std']]
# depth_mean = config_stats['D']['mean']
# depth_std = config_stats['D']['std']

# print(depth_mean)

        
# depth_transform_train = transforms.Compose([
#     transforms.Resize(300),
#     transforms.CenterCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(depth_mean,), std=(depth_std,))
# ])

# rgb_transform_train = transforms.Compose([
#     transforms.Resize(300),
#     transforms.RandomCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize(rgb_mean, rgb_std)
# ])

# rgb_transform_test = transforms.Compose([
#     transforms.Resize(300),
#     transforms.CenterCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize(rgb_mean, rgb_std)
# ])

# dataset = LocalBotDataset('seq6',depth_transform=depth_transform_train ,rgb_transform=rgb_transform_train, inputs=['depth_image', 'rgb_image'])

# for i in range(100,110):
#     print(f'depth size: {dataset[i][0].shape}')
#     print(f'rgb size: {dataset[i][1].shape}')
    
#     print(f'depth mean: {np.mean(dataset[i][0].numpy())}')
#     print(f'rgb mean: {np.mean(dataset[i][1].numpy())}')

