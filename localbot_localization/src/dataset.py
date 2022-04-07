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
        
        # load pose
        matrix = np.loadtxt(f'{self.path_seq}/frame-{index:05d}.pose.txt', delimiter=',')
        quaternion = matrixToQuaternion(matrix)
        quaternion = normalize_quat(quaternion)
        xyz = matrixToXYZ(matrix)
        pose = np.append(xyz, quaternion)

        # Transform to torch format
        point_set = torch.from_numpy(point_set.astype(np.float32))
        pose = torch.from_numpy(pose.astype(np.float32))

        return point_set, pose
                

    def __len__(self):
        return sum(f.endswith('pose.txt') for f in os.listdir(self.path_seq))
    
    def getConfig(self):
        with open(f'{self.path_seq}/config.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
        return config


#dataset = LocalBotDataset('seq_test_v')
#print(dataset[0][0].shape)
#print(sum(torch.isnan(dataset[78][0])))


class LocalBotDatasetDepth(data.Dataset):
    def __init__(self, path_seq, depth_normalizer=8.0):
        self.root = f'{os.environ["HOME"]}/datasets/localbot'
        self.seq = path_seq
        self.path_seq = f'{self.root}/{path_seq}'
        self.depth_normalizer = depth_normalizer

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
        cv_image = cv2.imread(f'{self.path_seq}/frame-{index:05d}.depth.png', cv2.IMREAD_UNCHANGED)
        cv_image = cv_image.astype(np.float32) / 1000.0  # to meters
        # normalize using max_depth_value
        # TODO test with different normalizations
        cv_image = cv_image/self.depth_normalizer

        # Convert to pytorch format
        h,w = cv_image.shape
        cv_image = cv_image.reshape((1,h,w))
        depth_image = torch.from_numpy(cv_image)

        return point_set, depth_image, pose

    def __len__(self):
        return sum(f.endswith('pose.txt') for f in os.listdir(self.path_seq))

    def getConfig(self):
        with open(f'{self.path_seq}/config.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
        return config

# dataset = LocalBotDataset('seq_test_v')
# print(dataset[0][0].shape)
# print(sum(torch.isnan(dataset[78][0])))

