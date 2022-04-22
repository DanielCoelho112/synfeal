import cv2
import torch.utils.data as data
from localbot_core.src.utilities import read_pcd, matrixToXYZ, matrixToQuaternion
from localbot_localization.src.utilities import normalize_quat
import numpy as np
import torch
import os
import yaml
from yaml.loader import SafeLoader

# pytorch datasets: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class LocalBotDataset(data.Dataset):
    def __init__(self, path_seq):
        self.root = f'{os.environ["HOME"]}/datasets/localbot'
        self.seq = path_seq
        self.path_seq = f'{self.root}/{path_seq}'
        
    def __getitem__(self, index):
        # return Nx3, 1x6 torch tensors
        # load point cloud
        pc_raw = read_pcd(f'{self.path_seq}/frame-{index:05d}.pcd')
        point_set = np.vstack([pc_raw.pc_data['x'], pc_raw.pc_data['y'], pc_raw.pc_data['z']]).T  # stays NX3
        point_set = torch.from_numpy(point_set.astype(np.float32))
        
        # load pose
        matrix = np.loadtxt(f'{self.path_seq}/frame-{index:05d}.pose.txt', delimiter=',')
        quaternion = matrixToQuaternion(matrix)
        quaternion = normalize_quat(quaternion)
        xyz = matrixToXYZ(matrix)
        pose = np.append(xyz, quaternion)
        pose = torch.from_numpy(pose.astype(np.float32))

        # load depth image
        cv_image = np.load(f'{self.path_seq}/frame-{index:05d}.depth.npy').astype(np.float32)        
        # Convert to pytorch format
        h,w = cv_image.shape
        cv_image = cv_image.reshape((1,h,w))
        depth_image = torch.from_numpy(cv_image)
        
        # load rgb image
        cv_image = np.load(f'{self.path_seq}/frame-{index:05d}.rgb.npy').astype(np.float32)        
        # Convert to pytorch format
        rgb_image = torch.from_numpy(cv_image)

        return point_set, depth_image, rgb_image, pose
    

    def __len__(self):
        return sum(f.endswith('pose.txt') for f in os.listdir(self.path_seq))
    
    def getConfig(self):
        with open(f'{self.path_seq}/config.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
        return config

    def setConfig(self, config):
        with open(f'{self.path_seq}/config.yaml', 'w') as f:
            yaml.dump(config, f)
        

#dataset = LocalBotDataset('seq_test_v')
#print(dataset[0][0].shape)
#print(sum(torch.isnan(dataset[78][0])))

